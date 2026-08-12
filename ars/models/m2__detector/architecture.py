from    typing              import Tuple, Any, Literal, Callable
from    dataclasses         import dataclass
from    functools           import reduce
from    itertools           import starmap, chain, tee

import  jax                 as jx
import  jax.numpy           as jp

from    flax                import linen as nn
from    flax.core           import FrozenDict
from    flax.linen          import initializers
from    flax.training       import train_state

import  optax               as ox

from    ars.models.metrics  import Loss, LossKind


type Array          = jp.ndarray
type Params         = Any
type Direction      = Literal['unidirectional', 'bidirectional']
type Activation     = Literal['relu', 'tanh']
type LayersArch[T]  = Tuple[Tuple[T, int], ...]


@dataclass(frozen = True)
class HyperParamsLSTMAE:
    sz_features     : int
    sz_latent       : int
    layers_arch     : LayersArch[Direction]
    dropout_rate    : float                                                                 = 0.0
    decoder_type    : Literal['autoregressive', 'linear']                                   = 'autoregressive'
    loss_type       : LossKind                                                              = 'mse'
    huber_delta     : float                                                                 = 1.0


@dataclass(frozen = True)
class HyperParamsFMLPAE:
    sz_latent_epi   : int
    sz_latent_sem   : int
    layers_arch     : LayersArch[Activation]
    dropout_rate    : float                                             = 0.0
    use_batch_norm  : bool                                              = False
    loss_type       : LossKind                                          = 'mse'
    huber_delta     : float                                             = 1.0


class TrainState(train_state.TrainState):
    batch_stats : Params    = None


@dataclass(frozen = True)
class Calibration:
    epi_median          : float
    epi_mad             : float
    sem_median          : float
    sem_mad             : float
    comb_median         : float
    comb_mad            : float
    comb_temperature    : float
    aux_z_anomaly       : float
    w_epi               : float
    w_sem               : float
    w_comb              : float
    bias                : float


@dataclass(frozen = True)
class Confidence:
    e_epi       : Array
    e_sem       : Array
    e_comb      : Array
    p_anomaly   : Array
    confidence  : Array
    is_anomaly  : Array


@dataclass(frozen = True)
class PreparedData:
    epi_dim : int
    sem_dim : int
    max_len : int

    train_epi_pad   : Array
    train_epi_mask  : Array
    train_sem_pad   : Array
    train_sem_mask  : Array
    train_labels    : Tuple[int, ...]

    val_epi_pad_normal  : Array
    val_epi_mask_normal : Array
    val_sem_pad_normal  : Array
    val_sem_mask_normal : Array
    val_epi_pad_mixed   : Array
    val_epi_mask_mixed  : Array
    val_sem_pad_mixed   : Array
    val_sem_mask_mixed  : Array
    val_labels_mixed    : Tuple[int, ...]

    test_epi_pad    : Array
    test_epi_mask   : Array
    test_sem_pad    : Array
    test_sem_mask   : Array
    test_labels     : Tuple[int, ...]


jx.tree_util.register_pytree_node(
    Confidence,
    lambda c: ((c.e_epi, c.e_sem, c.e_comb, c.p_anomaly, c.confidence, c.is_anomaly), None),
    lambda _aux, children: Confidence(*children))


@dataclass(frozen = True)
class LSTM:
    @staticmethod
    def make_lstm_cell(features: int, dtype: jp.dtype, param_dtype: jp.dtype) -> nn.OptimizedLSTMCell:
        return nn.OptimizedLSTMCell(
            features                = features,
            dtype                   = dtype,
            param_dtype             = param_dtype,
            kernel_init             = initializers.lecun_normal(),
            recurrent_kernel_init   = initializers.orthogonal(),
            bias_init               = initializers.zeros_init())

    @staticmethod
    def make_rnn_layer(layer_type: str, out_size: int, dtype: jp.dtype, param_dtype: jp.dtype, name: str) -> nn.Module:
        cell = LSTM.make_lstm_cell(out_size, dtype, param_dtype)
        match layer_type:
            case 'unidirectional': return nn.RNN(cell, return_carry = False, name = name)
            case 'bidirectional':
                forward     = nn.RNN(cell, return_carry = False, name = f'{name}_f')
                backward    = nn.RNN(cell, return_carry = False, name = f'{name}_b')
                return nn.Bidirectional(
                    forward, backward,
                    merge_fn    = lambda a, b: jp.concatenate((a, b), axis = -1),
                    name        = name)
            case _: raise ValueError(f'неизвестный тип слоя: {layer_type}')

    @staticmethod
    def make_encoder_layer(idx, ltype, out_size, dtype, param_dtype):
        return LSTM.make_rnn_layer(ltype, out_size, dtype, param_dtype, f'enc_{idx}')

    @staticmethod
    def make_decoder_layer(idx, out_size, dtype, param_dtype):
        return LSTM.make_rnn_layer('unidirectional', out_size, dtype, param_dtype, f'dec_{idx}')

    @staticmethod
    def make_encoder_proj(idx, ltype, out_size, layers_arch, dtype, param_dtype):
        next_out_size   = layers_arch[idx + 1][1] if idx < len(layers_arch) - 1 else None
        current_out     = out_size * (2 if ltype == 'bidirectional' else 1)

        return (nn.Dense(
                    next_out_size,
                    dtype       = dtype,
                    param_dtype = param_dtype,
                    kernel_init = initializers.lecun_normal(),
                    name        = f'proj_{idx}')
                if next_out_size is not None and current_out != next_out_size else None)


class LSTM_AE(nn.Module):
    hp          : HyperParamsLSTMAE
    dtype       : jp.dtype          = jp.float32
    param_dtype : jp.dtype          = jp.float32

    def setup(self):
        hp          = self.hp
        layers_arch = hp.layers_arch
        idxs        = range(len(layers_arch))

        rev_for_dec, rev_for_h, rev_for_c   = tee(reversed(layers_arch), 3)
        hidden_sizes_h                      = starmap(lambda t, d: d, rev_for_h)
        hidden_sizes_c                      = starmap(lambda t, d: d, rev_for_c)

        enc_specs_for_layers, enc_specs_for_proj = tee(zip(idxs, *zip(*layers_arch)), 2)
        self.encoder_layers = tuple(starmap(
            lambda i, ltype, out: LSTM.make_encoder_layer(i, ltype, out, self.dtype, self.param_dtype),
            enc_specs_for_layers))
        self.encoder_proj   = tuple(starmap(
            lambda i, ltype, out: LSTM.make_encoder_proj(i, ltype, out, layers_arch, self.dtype, self.param_dtype),
            enc_specs_for_proj))
        self.decoder_layers = tuple(starmap(
            lambda i, spec: LSTM.make_decoder_layer(i, spec[1], self.dtype, self.param_dtype),
            zip(idxs, rev_for_dec)))
        self.h0_projs = tuple(starmap(
            lambda i, sz: nn.Dense(sz,
                dtype       = self.dtype,
                param_dtype = self.param_dtype,
                kernel_init = initializers.lecun_normal(),
                name        = f'h0_proj_{i}'),
            enumerate(hidden_sizes_h)))
        self.c0_projs = tuple(starmap(
            lambda i, sz: nn.Dense(sz,
                dtype       = self.dtype,
                param_dtype = self.param_dtype,
                kernel_init = initializers.lecun_normal(),
                name        = f'c0_proj_{i}'),
            enumerate(hidden_sizes_c)))

        self.latent_proj    = nn.Dense(hp.sz_latent,
            dtype       = self.dtype,
            param_dtype = self.param_dtype,
            kernel_init = initializers.lecun_normal())
        self.out_proj       = nn.Dense(hp.sz_features,
            dtype       = self.dtype,
            param_dtype = self.param_dtype,
            kernel_init = initializers.lecun_normal())
        self.dropout = nn.Dropout(rate = hp.dropout_rate) if (hp.dropout_rate > 0.0) else None
        self.decoder_type = hp.decoder_type

    def encode(self, xs: Array, seq_lengths: None | Array = None, training: bool = True) -> Array:
        batch_size, max_len, _  = xs.shape
        lengths                 = seq_lengths if seq_lengths is not None else jp.full((batch_size,), max_len, dtype = jp.int32)

        def apply_layer(h, layer_proj):
            layer, proj = layer_proj
            outputs     = layer(h, seq_lengths = lengths, return_carry = False)
            outputs     = self.dropout(outputs, deterministic = not training) if self.dropout is not None else outputs
            return proj(outputs) if proj is not None else outputs

        h   = reduce(apply_layer, zip(self.encoder_layers, self.encoder_proj), xs)
        h   = h[jp.arange(batch_size), lengths - 1]
        return self.latent_proj(h)

    def decode_autoregressive(self, latent: Array, target_len: int, training: bool) -> Array:
        batch_size      = latent.shape[0]
        initial_states  = tuple(starmap(
            lambda c_proj, h_proj: (c_proj(latent), h_proj(latent)),
            zip(self.c0_projs, self.h0_projs)))
        initial_input   = jp.zeros((batch_size, 1, self.hp.sz_features), dtype = self.dtype)

        def step_fn(module, carry, _):
            x_prev, states = carry

            def layer_step(layer_acc, layer_state):
                x_in, new_states_acc = layer_acc
                layer, (c, h)           = layer_state
                (c_new, h_new), out     = layer(x_in, initial_carry = (c, h), return_carry = True)
                x_out = (module.dropout(out, deterministic = not training)
                        if module.dropout is not None else out)
                return x_out, chain(new_states_acc, ((c_new, h_new),))

            x_hidden, new_states_ch = reduce(layer_step, zip(module.decoder_layers, states), (x_prev, ()))
            x_pred                  = module.out_proj(x_hidden)
            return (x_pred, tuple(new_states_ch)), x_pred

        scanned     = nn.scan(
            step_fn,
            variable_broadcast  = 'params',
            split_rngs          = {'params': False, 'dropout': True},
            length              = target_len)
        _, outputs  = scanned(self, (initial_input, initial_states), None)
        return jp.transpose(jp.squeeze(outputs, axis = 2), (1, 0, 2))

    def decode_linear(self, latent: Array, target_len: int, training: bool) -> Array:
        batch_size      = latent.shape[0]
        zeros           = jp.zeros((batch_size, target_len, self.hp.sz_features), dtype = self.dtype)
        initial_states  = starmap(
            lambda c_proj, h_proj: (c_proj(latent), h_proj(latent)),
            zip(self.c0_projs, self.h0_projs))

        def step(inputs, layer_state):
            layer, (c0, h0) = layer_state
            outputs = layer(inputs, initial_carry = (c0, h0))
            return self.dropout(outputs, deterministic = not training) if self.dropout is not None else outputs

        outputs = reduce(step, zip(self.decoder_layers, initial_states), zeros)
        return self.out_proj(outputs)

    def decode(self, latent: Array, target_len: int, training: bool = True) -> Array:
        match self.decoder_type:
            case 'autoregressive': return self.decode_autoregressive(latent, target_len, training)
            case 'linear':         return self.decode_linear(latent, target_len, training)
            case _:                raise ValueError(f'неизвестный тип декодера: {self.decoder_type}')

    def __call__(self, xs: Array, seq_lengths: None | Array, training: bool) -> Tuple[Array, Array]:
        latent  = self.encode(xs, seq_lengths, training)
        recon   = self.decode(latent, xs.shape[1], training)
        return recon, latent


class FMLP_AE(nn.Module):
    hp          : HyperParamsFMLPAE
    dtype       : jp.dtype          = jp.float32
    param_dtype : jp.dtype          = jp.float32

    @nn.compact
    def __call__(self, epi_latent: Array, sem_latent: Array, training: bool) -> Array:
        hp              = self.hp
        in_size         = hp.sz_latent_epi + hp.sz_latent_sem
        xs              = jp.concatenate((epi_latent, sem_latent), axis = -1)
        hidden_sizes    = tuple(starmap(lambda act, dim: dim, hp.layers_arch))
        activations     = tuple(starmap(lambda act, dim: act, hp.layers_arch))

        def act_fn(name):
            match name:
                case 'relu': return nn.relu
                case 'tanh': return nn.tanh
                case _:      raise ValueError(f'неподдерживаемая активация: {name}')

        def make_ops(sizes):
            sizes_seq   = (in_size,) + sizes + (in_size,)
            pairs       = zip(range(len(sizes_seq) - 1), sizes_seq[1:])

            def layer_ops(idx, out_sz):
                head    = (('dense', out_sz),)
                body    = (((('batchnorm',),) if hp.use_batch_norm else ()) + (('act', act_fn(activations[idx])),) + (
                            (('dropout', hp.dropout_rate),) if hp.dropout_rate > 0.0 else ())) if idx < len(sizes_seq) - 2 else ()
                
                return head + body

            return chain.from_iterable(starmap(layer_ops, pairs))

        enc_ops = make_ops(hidden_sizes)
        dec_ops = make_ops(hidden_sizes[::-1])

        def apply_ops(x, ops):
            def step(x, op):
                match op:
                    case ('dense', sz):     return nn.Dense(sz, dtype = self.dtype, param_dtype = self.param_dtype)(x)
                    case ('act', f):        return f(x)
                    case ('dropout', rate): return nn.Dropout(rate = rate, deterministic = not training)(x)
                    case ('batchnorm',):    return nn.BatchNorm(use_running_average = not training)(x)
                    case _:                 return x
            return reduce(step, ops, x)

        return apply_ops(apply_ops(xs, enc_ops), dec_ops)


@dataclass(frozen = True)
class Models:
    epi_model       : LSTM_AE
    epi_state       : TrainState
    sem_model       : LSTM_AE
    sem_state       : TrainState
    combined_model  : FMLP_AE
    combined_state  : TrainState


@dataclass(frozen = True)
class InferenceMeta:
    max_len             : int
    best_threshold      : float
    normalize_latent    : bool
    epi_latent_mean     : Array
    epi_latent_std      : Array
    sem_latent_mean     : Array
    sem_latent_std      : Array
    calibration         : Calibration


@dataclass(frozen = True)
class LOSS:
    @staticmethod
    def masked_loss(recon: Array, target: Array, mask: Array, loss_type: LossKind = 'mse', huber_delta: float = 1.0) -> Array:
        return Loss.masked(recon, target, mask, loss_type, huber_delta, 1e-8)

    @staticmethod
    def compute_pointwise_loss(recon: Array, target: Array, loss_type: LossKind, huber_delta: float) -> Array:
        return Loss.pointwise(recon, target, loss_type, huber_delta)

    @staticmethod
    @jx.jit(static_argnames = ('model', 'training'))
    def apply_lstm_ae_loss(
            params      : Params,
            model       : LSTM_AE,
            xs          : Array,
            seq_lengths : Array,
            training    : bool = True,
    ) -> Tuple[Array, Array]:
        recon, latent = model.apply(FrozenDict({'params': params}), xs, seq_lengths, training)
        assert isinstance(latent, jp.ndarray)
        mask = jp.arange(xs.shape[1])[None, :] < seq_lengths[:, None]
        return LOSS.masked_loss(recon, xs, mask, model.hp.loss_type, model.hp.huber_delta), latent

    @staticmethod
    @jx.jit(static_argnames = ('model', 'training'))
    def apply_fmlp_ae_loss(
            params      : Params,
            model       : FMLP_AE,
            epi_latent  : Array,
            sem_latent  : Array,
            training    : bool = True,
            rngs        : None | dict[str, jx.Array] = None,
    ) -> Array:
        recon = model.apply(FrozenDict({'params': params}), epi_latent, sem_latent, training, rngs = rngs)
        assert isinstance(recon, jp.ndarray)
        target = jp.concatenate((epi_latent, sem_latent), axis = -1)
        return Loss.mean(recon, target, model.hp.loss_type, model.hp.huber_delta)


@dataclass(frozen = True)
class TRAIN:
    @staticmethod
    def make_train_state(
            rng           : jx.Array,
            model         : nn.Module,
            learning_rate : float,
            weight_decay  : float = 0.0,
            clip_grad     : float = 1.0,
            schedule_fn   : None | Callable = None,
            input_shape   : None | Tuple[int, ...] = None,
    ) -> TrainState:
        match model:
            case LSTM_AE():
                if input_shape is None:
                    raise ValueError('для LSTM_AE необходимо указать input_shape')
                dummy_input         = jp.ones((1,) + input_shape, dtype = jp.float32)
                dummy_seq_lengths   = jp.full((1,), input_shape[0], dtype = jp.int32)
                variables           = model.init(rng, dummy_input, dummy_seq_lengths, training = True)
                params              = variables['params']
                batch_stats         = variables.get('batch_stats', {})
            case FMLP_AE():
                epi_dummy   = jp.ones((1, model.hp.sz_latent_epi))
                sem_dummy   = jp.ones((1, model.hp.sz_latent_sem))
                variables   = model.init(rng, epi_dummy, sem_dummy, training = True)
                params      = variables['params']
                batch_stats = variables.get('batch_stats', {})
            case _: raise TypeError('неизвестный тип модели')

        schedule    = schedule_fn if schedule_fn is not None else ox.constant_schedule(learning_rate)
        optimizer   = ox.chain(
            ox.clip_by_global_norm(clip_grad),
            ox.adamw(learning_rate = schedule, weight_decay = weight_decay))
        return TrainState.create(apply_fn = model.apply, params = params, tx = optimizer, batch_stats = batch_stats)

    @staticmethod
    @jx.jit(static_argnames = ('training', 'loss_type', 'huber_delta'))
    def train_step_lstm_ae(
            state       : TrainState,
            xs          : Array,
            seq_lengths : Array,
            training    : bool = True,
            rng         : None | jx.Array = None,
            loss_type   : LossKind = 'mse',
            huber_delta : float = 1.0,
    ) -> Tuple[TrainState, Array]:
        def loss_fn(params):
            rngs            = {'dropout': rng} if (training and rng is not None) else None
            recon, _latent  = state.apply_fn({'params': params}, xs, seq_lengths, training, rngs = rngs)
            mask            = jp.arange(xs.shape[1])[None, :] < seq_lengths[:, None]
            return LOSS.masked_loss(recon, xs, mask, loss_type, huber_delta)

        loss, grads = jx.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads = grads), loss

    @staticmethod
    @jx.jit(static_argnames = ('training', 'loss_type', 'huber_delta'))
    def train_step_fmlp_ae(
            state       : TrainState,
            epi_latent  : Array,
            sem_latent  : Array,
            training    : bool = True,
            rng         : None | jx.Array = None,
            loss_type   : LossKind = 'mse',
            huber_delta : float = 1.0,
    ) -> Tuple[TrainState, Array]:
        def loss_fn(params):
            rngs            = {'dropout': rng} if (training and rng is not None) else None
            variables       = {'params': params, 'batch_stats': state.batch_stats}
            recon, new_vars = state.apply_fn(
                variables, epi_latent, sem_latent, training, rngs = rngs, mutable = ['batch_stats'])
            target          = jp.concatenate((epi_latent, sem_latent), axis = -1)
            return Loss.mean(recon, target, loss_type, huber_delta), new_vars

        (loss, new_vars), grads = jx.value_and_grad(loss_fn, has_aux = True)(state.params)
        updated = state.apply_gradients(grads = grads)
        return updated.replace(batch_stats = new_vars['batch_stats']), loss

    @staticmethod
    @jx.jit(static_argnames = ('model',))
    def lstm_ae_encode_batch(params: Params, model: LSTM_AE, xs: Array, seq_lengths: None | Array = None) -> Array:
        result = model.apply(FrozenDict({'params': params}), xs, seq_lengths, method = model.encode, training = False)
        assert isinstance(result, jp.ndarray)
        return result

    @staticmethod
    @jx.jit(static_argnames = ('model',))
    def lstm_ae_sse_count(params: Params, model: LSTM_AE, xs: Array, seq_lengths: Array, mask: Array) -> Tuple[Array, Array]:
        recon, _    = model.apply(FrozenDict({'params': params}), xs, seq_lengths, training = False)
        mask_exp    = mask[..., None]
        return jp.sum((recon - xs) ** 2 * mask_exp), jp.sum(mask_exp)

    @staticmethod
    @jx.jit(static_argnames = ('model',))
    def lstm_ae_decode_batch(params: Params, model: LSTM_AE, latent: Array, target_len: int) -> Array:
        result = model.apply(FrozenDict({'params': params}), latent, target_len, method = model.decode, training = False)
        assert isinstance(result, jp.ndarray)
        return result

    @staticmethod
    @jx.jit(static_argnames = ('model', 'training'))
    def fmlp_ae_forward(state: TrainState, model: FMLP_AE, epi_latent: Array, sem_latent: Array, training: bool = False) -> Array:
        variables   = FrozenDict({'params': state.params, 'batch_stats': state.batch_stats})
        result      = model.apply(variables, epi_latent, sem_latent, training)
        assert isinstance(result, jp.ndarray)
        return result


@dataclass(frozen = True)
class Branch:
    @staticmethod
    def encode(model: LSTM_AE, state: TrainState, padded: Array, mask: Array, chunk: int = 1024) -> Array:
        seq_lengths = jp.sum(mask, axis = 1).astype(jp.int32)
        bounds      = range(0, padded.shape[0], chunk)

        return jp.concatenate(tuple(map(
            lambda i: TRAIN.lstm_ae_encode_batch(state.params, model, padded[i:i + chunk], seq_lengths[i:i + chunk]),
            bounds)), axis = 0)

    @staticmethod
    def lstm_recon_mse(model: LSTM_AE, state: TrainState, padded: Array, mask: Array, chunk: int = 1024) -> Array:
        bounds      = range(0, padded.shape[0], chunk)
        sse, cnt    = map(jp.stack, zip(*map(
            lambda i: TRAIN.lstm_ae_sse_count(state.params, model, padded[i:i + chunk],
                jp.sum(mask[i:i + chunk], axis = 1).astype(jp.int32), mask[i:i + chunk]),
            bounds)))

        return sse.sum() / cnt.sum()

    @staticmethod
    def combined_recon_mse(state: TrainState, model: FMLP_AE, epi_lat: Array, sem_lat: Array) -> Array:
        recon   = TRAIN.fmlp_ae_forward(state, model, epi_lat, sem_lat, training = False)
        target  = jp.concatenate((epi_lat, sem_lat), axis = -1)
        return jp.mean((recon - target) ** 2)

    @staticmethod
    def combined_errors(state: TrainState, model: FMLP_AE, epi_lat: Array, sem_lat: Array) -> Array:
        recon       = TRAIN.fmlp_ae_forward(state, model, epi_lat, sem_lat, training = False)
        target      = jp.concatenate((epi_lat, sem_lat), axis = -1)
        pointwise   = Loss.pointwise(recon, target, model.hp.loss_type, model.hp.huber_delta)
        return jp.mean(pointwise, axis = -1)

    @staticmethod
    def params_norm(state: TrainState) -> Array:
        leaves = jx.tree_util.tree_leaves(state.params)
        return jp.sqrt(reduce(lambda acc, p: acc + jp.sum(p ** 2), leaves, jp.zeros((), dtype = jp.float32)))