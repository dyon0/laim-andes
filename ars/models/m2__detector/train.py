from    typing                                  import Tuple, Callable, cast
from    dataclasses                             import dataclass
from    functools                               import reduce, partial
from    math                                    import isfinite

import  jax                                     as jx
import  jax.numpy                               as jp

from    flax.core                               import FrozenDict

from    ars.models.m2__detector.architecture    import (
    LSTM_AE, FMLP_AE, TRAIN, LOSS,
    TrainState, Array, Params, LossKind)


@dataclass(frozen = True)
class Trainer:
    @staticmethod
    @partial(jx.jit, static_argnames = ('model',))
    def _compute_val_loss_lstm(params: Params, model: LSTM_AE, val_padded: Array, val_mask: Array) -> Array:
        seq_lengths = jp.sum(val_mask, axis = 1).astype(jp.int32)
        loss, _     = LOSS.apply_lstm_ae_loss(params, model, val_padded, seq_lengths, training = False)
        return loss

    @staticmethod
    @partial(jx.jit, static_argnames = ('model', 'batch_size'))
    def _compute_val_loss_lstm_batched(
            params      : Params,
            model       : LSTM_AE,
            val_padded  : Array,
            val_mask    : Array,
            batch_size  : int,
    ) -> Array:
        num_samples     = val_padded.shape[0]
        pad_n           = (-num_samples) % batch_size
        val_padded_p    = jp.pad(val_padded, ((0, pad_n), (0, 0), (0, 0)), constant_values = 0.0)
        val_mask_p      = jp.pad(val_mask,   ((0, pad_n), (0, 0)),         constant_values = False)
        n_batches       = val_padded_p.shape[0] // batch_size
        val_b           = val_padded_p.reshape(n_batches, batch_size, *val_padded.shape[1:])
        mask_b          = val_mask_p  .reshape(n_batches, batch_size, val_mask.shape[1])

        def _step(acc, batched):
            batch_x, batch_mask = batched
            sum_loss, sum_count = acc
            seq_lens            = jp.sum(batch_mask, axis = 1).astype(jp.int32)
            recon, _            = model.apply(FrozenDict({'params': params}), batch_x, seq_lens, training = False)
            mask3d              = batch_mask[..., None].astype(jp.float32)
            pointwise           = LOSS.compute_pointwise_loss(recon, batch_x, model.hp.loss_type, model.hp.huber_delta)

            # F-23: per-element normalization (see Loss.masked)
            return (sum_loss + jp.sum(pointwise * mask3d), sum_count + jp.sum(mask3d) * recon.shape[-1]), None

        zero_acc = (jp.zeros((), dtype = jp.float32), jp.zeros((), dtype = jp.float32))
        (total_loss, total_count), _ = jx.lax.scan(_step, zero_acc, (val_b, mask_b))

        return total_loss / (total_count + 1e-8) #todo: вынести eps в конфиг

    @staticmethod
    @partial(jx.jit, static_argnames = ('model',))
    def _compute_val_loss_fmlp(state: TrainState, model: FMLP_AE, val_epi: Array, val_sem: Array) -> Array:
        variables   = FrozenDict({'params': state.params, 'batch_stats': state.batch_stats})
        recon       = cast(Array, model.apply(variables, val_epi, val_sem, training = False))
        target      = jp.concatenate((val_epi, val_sem), axis = -1)

        return jp.mean(LOSS.compute_pointwise_loss(recon, target, model.hp.loss_type, model.hp.huber_delta))

    @staticmethod
    @partial(jx.jit, static_argnames = ('model', 'batch_size'))
    def _compute_val_loss_fmlp_batched(
            state       : TrainState,
            model       : FMLP_AE,
            val_epi     : Array,
            val_sem     : Array,
            batch_size  : int,
    ) -> Array:
        num_samples = val_epi.shape[0]
        pad_n       = (-num_samples) % batch_size
        val_epi_p   = jp.pad(val_epi, ((0, pad_n), (0, 0)), constant_values = 0.0)
        val_sem_p   = jp.pad(val_sem, ((0, pad_n), (0, 0)), constant_values = 0.0)
        valid_mask  = jp.concatenate((
            jp.ones ((num_samples,), dtype = jp.float32),
            jp.zeros((pad_n,),       dtype = jp.float32)))
        n_batches   = val_epi_p.shape[0] // batch_size
        epi_b       = val_epi_p .reshape(n_batches, batch_size, val_epi.shape[1])
        sem_b       = val_sem_p .reshape(n_batches, batch_size, val_sem.shape[1])
        m_b         = valid_mask.reshape(n_batches, batch_size)
        variables   = FrozenDict({'params': state.params, 'batch_stats': state.batch_stats})

        def _step(acc, batched):
            epi_batch, sem_batch, m = batched
            sum_loss, sum_count     = acc
            recon                   = cast(Array, model.apply(variables, epi_batch, sem_batch, training = False))
            target                  = jp.concatenate((epi_batch, sem_batch), axis = -1)
            pointwise               = LOSS.compute_pointwise_loss(recon, target, model.hp.loss_type, model.hp.huber_delta)
            per_sample              = jp.mean(pointwise, axis = -1)

            return (sum_loss + jp.sum(per_sample * m), sum_count + jp.sum(m)), None

        zero_acc = (jp.zeros((), dtype = jp.float32), jp.zeros((), dtype = jp.float32))
        (total_loss, total_count), _ = jx.lax.scan(_step, zero_acc, (epi_b, sem_b, m_b))

        return total_loss / (total_count + 1e-8)

    @staticmethod
    @partial(jx.jit, static_argnames = ('batch_size', 'loss_type', 'huber_delta'))
    def _run_lstm_epoch(
            state           : TrainState,
            rng             : jx.Array,
            train_padded    : Array,
            train_mask      : Array,
            batch_size      : int,
            loss_type       : LossKind,
            huber_delta     : float,
    ) -> Tuple[TrainState, Array, jx.Array]:
        num_samples             = train_padded.shape[0]
        n_batches               = num_samples // batch_size
        usable                  = n_batches * batch_size
        rng, perm_rng, scan_rng = jx.random.split(rng, 3)
        perm                    = jx.random.permutation(perm_rng, num_samples)
        train_b                 = train_padded[perm[:usable]].reshape(n_batches, batch_size, *train_padded.shape[1:])
        mask_b                  = train_mask  [perm[:usable]].reshape(n_batches, batch_size, train_mask.shape[1])
        rngs_b                  = jx.random.split(scan_rng, n_batches)

        def _batch_step(state, batched):
            batch_x, batch_mask, step_rng   = batched
            seq_lengths                     = jp.sum(batch_mask, axis = 1).astype(jp.int32)

            return TRAIN.train_step_lstm_ae(
                state, batch_x, seq_lengths,
                training = True, rng = step_rng, loss_type = loss_type, huber_delta = huber_delta)

        final_state, batch_losses = jx.lax.scan(_batch_step, state, (train_b, mask_b, rngs_b))

        return final_state, jp.mean(batch_losses), rng

    @staticmethod
    @partial(jx.jit, static_argnames = ('batch_size', 'loss_type', 'huber_delta'))
    def _run_fmlp_epoch(
            state       : TrainState,
            rng         : jx.Array,
            train_epi   : Array,
            train_sem   : Array,
            batch_size  : int,
            loss_type   : LossKind,
            huber_delta : float,
    ) -> Tuple[TrainState, Array, jx.Array]:
        num_samples             = train_epi.shape[0]
        n_batches               = num_samples // batch_size
        usable                  = n_batches * batch_size
        rng, perm_rng, scan_rng = jx.random.split(rng, 3)
        perm                    = jx.random.permutation(perm_rng, num_samples)
        epi_b                   = train_epi[perm[:usable]].reshape(n_batches, batch_size, train_epi.shape[1])
        sem_b                   = train_sem[perm[:usable]].reshape(n_batches, batch_size, train_sem.shape[1])
        rngs_b                  = jx.random.split(scan_rng, n_batches)

        def _batch_step(state, batched):
            epi_batch, sem_batch, step_rng = batched
            return TRAIN.train_step_fmlp_ae(
                state, epi_batch, sem_batch,
                training = True, rng = step_rng, loss_type = loss_type, huber_delta = huber_delta)

        final_state, batch_losses = jx.lax.scan(_batch_step, state, (epi_b, sem_b, rngs_b))

        return final_state, jp.mean(batch_losses), rng

    @staticmethod
    def _train_loop(
            initial_state   : TrainState,
            rng             : jx.Array,
            run_epoch       : Callable[[TrainState, jx.Array], Tuple[TrainState, Array, jx.Array]],
            compute_val     : Callable[[TrainState], Array],
            num_epochs      : int,
            patience        : int,
            target_loss     : None | float,
            label           : str,
    ) -> Tuple[TrainState, Tuple[float, ...]]:
        max_epochs  = 10000 if target_loss is not None else num_epochs #todo: вынести hard-cup в конфиг
        initial_acc = (initial_state, initial_state, float('inf'), 0, (), rng, False)

        def _do_epoch(acc, epoch_idx):
            state, best_state, best_val, no_imp, losses, rng, _ = acc
            state, avg_loss, rng                                = run_epoch(state, rng)
            val_loss                                            = float(compute_val(state))
            avg_loss_py                                         = float(avg_loss)
            # F-21: a non-finite loss must abort loudly, not train on garbage
            if not (isfinite(avg_loss_py) and isfinite(val_loss)):
                raise FloatingPointError(
                    f'{label}эпоха {epoch_idx + 1}: невалидная ошибка '
                    f'(train={avg_loss_py}, val={val_loss}) — обучение прервано')
            print(f'Epoch {epoch_idx + 1}, {label}Loss: {avg_loss_py:.6f}, Val Loss: {val_loss:.6f}')
            improved        = val_loss < best_val
            new_best_state  = state    if improved else best_state
            new_best_val    = val_loss if improved else best_val
            new_no_imp      = 0        if improved else (no_imp + 1)
            target_reached  = (target_loss is not None) and (val_loss <= target_loss)
            out_of_patience = new_no_imp >= patience
            new_stopped     = target_reached or out_of_patience
            if target_reached:
                print(f'Целевая ошибка {target_loss} достигнута. Остановка на эпохе {epoch_idx + 1}')
            elif out_of_patience:
                print(f'Нет улучшений за {patience} эпох. Остановка на эпохе {epoch_idx + 1}')
            return (state, new_best_state, new_best_val, new_no_imp, losses + (avg_loss_py,), rng, new_stopped)

        _step                                           = lambda acc, ei: acc if acc[-1] else _do_epoch(acc, ei)
        final_acc                                       = reduce(_step, range(max_epochs), initial_acc)
        _, final_best_state, _, _, final_losses, _, _   = final_acc

        return final_best_state, final_losses

    @staticmethod
    def train_lstm_ae(
            model           : LSTM_AE,
            train_padded    : Array,
            train_mask      : Array,
            val_padded      : Array,
            val_mask        : Array,
            learning_rate   : float,
            num_epochs      : int,
            batch_size      : int,
            rng             : jx.Array,
            input_shape     : Tuple[int, ...],
            weight_decay    : float,
            clip_grad       : float,
            schedule_fn     : None | Callable,
            patience        : int,
            target_loss     : None | float,
            val_batched     : bool = False,
    ) -> Tuple[TrainState, Tuple[float, ...]]:
        # F-40: batch_size > n_train means zero full batches per epoch; the
        # legacy code then trained on nothing (NaN losses) and silently kept
        # the random init as the "best" model — which was ranked and shipped.
        if train_padded.shape[0] < batch_size:
            raise ValueError(
                f'batch_size={batch_size} больше числа обучающих трасс '
                f'({train_padded.shape[0]}): эпоха не содержит ни одного батча. '
                f'Уменьшите batch_size эксперимента или увеличьте корпус.')
        state       = TRAIN.make_train_state(
            rng, model, learning_rate,
            weight_decay = weight_decay, clip_grad = clip_grad, schedule_fn = schedule_fn, input_shape = input_shape)
        run_epoch   = lambda s, r: Trainer._run_lstm_epoch(
            s, r, train_padded, train_mask, batch_size, model.hp.loss_type, model.hp.huber_delta)
        compute_val = ((lambda s: Trainer._compute_val_loss_lstm_batched(s.params, model, val_padded, val_mask, batch_size))
                       if val_batched else
                       (lambda s: Trainer._compute_val_loss_lstm(s.params, model, val_padded, val_mask)))
        
        return Trainer._train_loop(
            initial_state   = state, rng = rng, run_epoch = run_epoch, compute_val = compute_val,
            num_epochs      = num_epochs, patience = patience, target_loss = target_loss, label = '')

    @staticmethod
    def train_fmlp_ae(
            model           : FMLP_AE,
            train_epi       : Array,
            train_sem       : Array,
            val_epi         : Array,
            val_sem         : Array,
            learning_rate   : float,
            num_epochs      : int,
            batch_size      : int,
            rng             : jx.Array,
            weight_decay    : float,
            clip_grad       : float,
            schedule_fn     : None | Callable,
            patience        : int,
            target_loss     : None | float,
            val_batched     : bool = False,
    ) -> Tuple[TrainState, Tuple[float, ...]]:
        if train_epi.shape[0] < batch_size:  # F-40, see train_lstm_ae
            raise ValueError(
                f'batch_size={batch_size} больше числа обучающих трасс '
                f'({train_epi.shape[0]}): эпоха не содержит ни одного батча. '
                f'Уменьшите batch_size эксперимента или увеличьте корпус.')
        state       = TRAIN.make_train_state(
            rng, model, learning_rate,
            weight_decay = weight_decay, clip_grad = clip_grad, schedule_fn = schedule_fn, input_shape = None)
        run_epoch   = lambda s, r: Trainer._run_fmlp_epoch(
            s, r, train_epi, train_sem, batch_size, model.hp.loss_type, model.hp.huber_delta)
        compute_val = ((lambda s: Trainer._compute_val_loss_fmlp_batched(s, model, val_epi, val_sem, batch_size))
                       if val_batched else
                       (lambda s: Trainer._compute_val_loss_fmlp(s, model, val_epi, val_sem)))
        
        return Trainer._train_loop(
            initial_state   = state, rng = rng, run_epoch = run_epoch, compute_val = compute_val,
            num_epochs      = num_epochs, patience = patience, target_loss = target_loss, label = 'Combined ')