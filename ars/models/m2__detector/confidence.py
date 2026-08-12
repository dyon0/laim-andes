from    typing                                  import Tuple, cast
from    dataclasses                             import dataclass
from    functools                               import partial

import  jax                                     as jx
import  jax.numpy                               as jp

from    jax.nn                                  import sigmoid
from    jax.scipy.optimize                      import minimize

from    flax.core                               import FrozenDict

from    ars.models.m2__detector.architecture    import (
    LSTM_AE, FMLP_AE, LOSS,
    TrainState, Branch, Calibration, Confidence, Models, InferenceMeta, PreparedData, Array, Params)


@dataclass(frozen = True)
class Calibrate:
    @staticmethod
    @partial(jx.jit, static_argnames = ('model',))
    def lstm_branch(model: LSTM_AE, params: Params, padded: Array, mask: Array) -> Tuple[Array, Array]:
        seq_lengths     = jp.sum(mask, axis = 1).astype(jp.int32)
        recon, latent   = cast(Tuple[Array, Array], model.apply(FrozenDict({'params': params}), padded, seq_lengths, training = False))
        pointwise       = LOSS.compute_pointwise_loss(recon, padded, model.hp.loss_type, model.hp.huber_delta)
        mask_3d         = mask[..., None].astype(jp.float32)
        per_sample_sum  = jp.sum(pointwise * mask_3d, axis = (1, 2))
        # F-23: per-element normalization — errors comparable across branches
        per_sample_cnt  = jp.sum(mask_3d, axis = (1, 2)) * padded.shape[-1]

        return per_sample_sum / (per_sample_cnt + 1e-8), latent

    @staticmethod
    def robust_stats(errors: Array) -> Tuple[Array, Array]:
        median  = jp.median(errors)
        mad     = jp.median(jp.abs(errors - median))
        
        return median, mad

    @staticmethod
    def fit_weights(logits_epi: Array, logits_sem: Array, logits_comb: Array, labels: Array) -> Tuple[Array, Array, Array, Array]:
        n_pos   = float(jp.sum(labels))
        n_neg   = float(labels.shape[0] - n_pos)
        priors  = jp.asarray((0.2, 0.2, 0.6, 0.0), dtype = jp.float32)
        if (n_pos < 1.0) or (n_neg < 1.0):
            return priors[0], priors[1], priors[2], priors[3]
        n_total         = n_pos + n_neg
        w_pos           = jp.asarray(n_total / (2.0 * n_pos), dtype = jp.float32)
        w_neg           = jp.asarray(n_total / (2.0 * n_neg), dtype = jp.float32)
        y               = labels.astype(jp.float32)
        sample_weight   = y * w_pos + (1.0 - y) * w_neg
        features        = jp.stack((logits_epi, logits_sem, logits_comb, jp.ones_like(logits_epi)), axis = 1)

        def neg_log_likelihood(params: Array) -> Array:
            z           = features @ params
            per_sample  = jp.maximum(z, 0.0) - z * y + jp.log1p(jp.exp(-jp.abs(z)))
            return jp.sum(per_sample * sample_weight) / jp.sum(sample_weight)

        init    = jp.zeros(4, dtype = jp.float32)
        result  = minimize(neg_log_likelihood, init, method = 'BFGS', options = {'maxiter': 200})
        return result.x[0], result.x[1], result.x[2], result.x[3]

    @staticmethod
    def compute(
            models              : Models,
            prepared            : PreparedData,
            best_threshold      : float,
            normalize_latent    : bool,
            epi_mean            : None | Array,
            epi_std             : None | Array,
            sem_mean            : None | Array,
            sem_std             : None | Array,
            eps                 : float,
    ) -> Calibration:
        train_epi_errs, train_epi_lat   = Calibrate.lstm_branch(
            models.epi_model, models.epi_state.params, prepared.train_epi_pad, prepared.train_epi_mask)
        train_sem_errs, train_sem_lat   = Calibrate.lstm_branch(
            models.sem_model, models.sem_state.params, prepared.train_sem_pad, prepared.train_sem_mask)
        if normalize_latent:
            train_epi_lat   = (train_epi_lat - epi_mean) / epi_std
            train_sem_lat   = (train_sem_lat - sem_mean) / sem_std
        train_comb_errs = Branch.combined_errors(models.combined_state, models.combined_model, train_epi_lat, train_sem_lat)

        epi_median, epi_mad     = Calibrate.robust_stats(train_epi_errs)
        sem_median, sem_mad     = Calibrate.robust_stats(train_sem_errs)
        comb_median, comb_mad   = Calibrate.robust_stats(train_comb_errs)

        val_epi_errs, val_epi_lat_m = Calibrate.lstm_branch(
            models.epi_model, models.epi_state.params, prepared.val_epi_pad_mixed, prepared.val_epi_mask_mixed)
        val_sem_errs, val_sem_lat_m = Calibrate.lstm_branch(
            models.sem_model, models.sem_state.params, prepared.val_sem_pad_mixed, prepared.val_sem_mask_mixed)
        if normalize_latent:
            val_epi_lat_m   = (val_epi_lat_m - epi_mean) / epi_std
            val_sem_lat_m   = (val_sem_lat_m - sem_mean) / sem_std
        val_comb_errs   = Branch.combined_errors(models.combined_state, models.combined_model, val_epi_lat_m, val_sem_lat_m)
        val_labels      = jp.asarray(prepared.val_labels_mixed, dtype = jp.int32)

        norm_mask                   = val_labels == 0
        anom_mask                   = val_labels == 1
        norm_count                  = jp.sum(norm_mask)
        anom_count                  = jp.sum(anom_mask)
        eps_safe                    = jp.asarray(1e-8, dtype = jp.float32)
        aux_z_anomaly               = jp.asarray(2.0,  dtype = jp.float32)
        fallback_T                  = jp.maximum(1.4826 * comb_mad, eps)
        mean_norm                   = jp.sum(jp.where(norm_mask, val_comb_errs, 0.0)) / jp.maximum(norm_count, 1)
        mean_anom                   = jp.sum(jp.where(anom_mask, val_comb_errs, 0.0)) / jp.maximum(anom_count, 1)
        var_norm                    = jp.sum(jp.where(norm_mask, (val_comb_errs - mean_norm) ** 2, 0.0)) / jp.maximum(norm_count, 1)
        var_anom                    = jp.sum(jp.where(anom_mask, (val_comb_errs - mean_anom) ** 2, 0.0)) / jp.maximum(anom_count, 1)
        pooled_sigma                = jp.sqrt((var_norm + var_anom) / 2.0)
        has_both                    = (norm_count > 1) & (anom_count > 1)
        comb_T                      = jp.maximum(jp.where(has_both, jp.maximum(pooled_sigma, fallback_T), fallback_T), eps_safe)
        z_epi_val                   = (val_epi_errs - epi_median) / (1.4826 * epi_mad + eps_safe)
        z_sem_val                   = (val_sem_errs - sem_median) / (1.4826 * sem_mad + eps_safe)
        logits_epi_val              = z_epi_val - aux_z_anomaly
        logits_sem_val              = z_sem_val - aux_z_anomaly
        logits_comb_val             = (val_comb_errs - best_threshold) / (comb_T + eps_safe)
        w_epi, w_sem, w_comb, bias  = Calibrate.fit_weights(logits_epi_val, logits_sem_val, logits_comb_val, val_labels)

        return Calibration(
            epi_median          = float(epi_median),
            epi_mad             = float(epi_mad),
            sem_median          = float(sem_median),
            sem_mad             = float(sem_mad),
            comb_median         = float(comb_median),
            comb_mad            = float(comb_mad),
            comb_temperature    = float(comb_T),
            aux_z_anomaly       = float(aux_z_anomaly),
            w_epi               = float(w_epi),
            w_sem               = float(w_sem),
            w_comb              = float(w_comb),
            bias                = float(bias))


@dataclass(frozen = True)
class Predict:
    @staticmethod
    @partial(jx.jit, static_argnames = ('epi_model', 'sem_model', 'combined_model'))
    def from_branches(
            epi_model       : LSTM_AE,
            epi_params      : Params,
            sem_model       : LSTM_AE,
            sem_params      : Params,
            combined_model  : FMLP_AE,
            combined_state  : TrainState,
            epi_padded      : Array,
            epi_mask        : Array,
            sem_padded      : Array,
            sem_mask        : Array,
            epi_lat_mean    : Array,
            epi_lat_std     : Array,
            sem_lat_mean    : Array,
            sem_lat_std     : Array,
            best_threshold  : Array,
            cal_epi_median  : Array,
            cal_epi_mad     : Array,
            cal_sem_median  : Array,
            cal_sem_mad     : Array,
            cal_comb_T      : Array,
            cal_aux_z       : Array,
            cal_w_epi       : Array,
            cal_w_sem       : Array,
            cal_w_comb      : Array,
            cal_bias        : Array,
    ) -> Confidence:
        e_epi, epi_lat  = Calibrate.lstm_branch(epi_model, epi_params, epi_padded, epi_mask)
        e_sem, sem_lat  = Calibrate.lstm_branch(sem_model, sem_params, sem_padded, sem_mask)
        epi_lat_n       = (epi_lat - epi_lat_mean) / epi_lat_std
        sem_lat_n       = (sem_lat - sem_lat_mean) / sem_lat_std
        e_comb          = Branch.combined_errors(combined_state, combined_model, epi_lat_n, sem_lat_n)
        eps_safe        = jp.asarray(1e-8, dtype = jp.float32)
        z_epi           = (e_epi - cal_epi_median) / (1.4826 * cal_epi_mad + eps_safe)
        z_sem           = (e_sem - cal_sem_median) / (1.4826 * cal_sem_mad + eps_safe)
        logit_epi       = z_epi - cal_aux_z
        logit_sem       = z_sem - cal_aux_z
        logit_comb      = (e_comb - best_threshold) / (cal_comb_T + eps_safe)
        logit_agg       = cal_w_epi * logit_epi + cal_w_sem * logit_sem + cal_w_comb * logit_comb + cal_bias
        p_anom          = sigmoid(logit_agg)
        return Confidence(
            e_epi       = e_epi,
            e_sem       = e_sem,
            e_comb      = e_comb,
            p_anomaly   = p_anom,
            confidence  = jp.maximum(p_anom, 1.0 - p_anom),
            is_anomaly  = e_comb > best_threshold)

    @staticmethod
    def batch(meta: InferenceMeta, models: Models, epi_padded: Array, epi_mask: Array, sem_padded: Array, sem_mask: Array) -> Confidence:
        cal = meta.calibration
        return Predict.from_branches(
            epi_model       = models.epi_model,
            epi_params      = models.epi_state.params,
            sem_model       = models.sem_model,
            sem_params      = models.sem_state.params,
            combined_model  = models.combined_model,
            combined_state  = models.combined_state,
            epi_padded      = epi_padded,
            epi_mask        = epi_mask,
            sem_padded      = sem_padded,
            sem_mask        = sem_mask,
            epi_lat_mean    = meta.epi_latent_mean,
            epi_lat_std     = meta.epi_latent_std,
            sem_lat_mean    = meta.sem_latent_mean,
            sem_lat_std     = meta.sem_latent_std,
            best_threshold  = jp.asarray(meta.best_threshold,   dtype = jp.float32),
            cal_epi_median  = jp.asarray(cal.epi_median,        dtype = jp.float32),
            cal_epi_mad     = jp.asarray(cal.epi_mad,           dtype = jp.float32),
            cal_sem_median  = jp.asarray(cal.sem_median,        dtype = jp.float32),
            cal_sem_mad     = jp.asarray(cal.sem_mad,           dtype = jp.float32),
            cal_comb_T      = jp.asarray(cal.comb_temperature,  dtype = jp.float32),
            cal_aux_z       = jp.asarray(cal.aux_z_anomaly,     dtype = jp.float32),
            cal_w_epi       = jp.asarray(cal.w_epi,             dtype = jp.float32),
            cal_w_sem       = jp.asarray(cal.w_sem,             dtype = jp.float32),
            cal_w_comb      = jp.asarray(cal.w_comb,            dtype = jp.float32),
            cal_bias        = jp.asarray(cal.bias,              dtype = jp.float32))