from    typing              import Tuple, Dict, Any, Literal, Callable
from    dataclasses         import dataclass
from    functools           import reduce

import  jax                 as jx
import  jax.numpy           as jp

from    flax                import linen as nn
from    flax.linen          import initializers

import  optax               as ox

from    ars.models.metrics  import Loss


type Array  = jp.ndarray
type Key    = jp.ndarray
type Params = Any

Kind = Literal['logreg', 'mlp', 'svm', 'kan', 'forest', 'boosting']


@dataclass(frozen = True)
class HyperLogReg:
    learning_rate   : float = 1e-2
    epochs          : int   = 200
    batch_size      : int   = 512
    weight_decay    : float = 1e-4


@dataclass(frozen = True)
class HyperMLP:
    hidden          : Tuple[int, ...]                   = (128, 64)
    activation      : Literal['relu', 'tanh', 'gelu']   = 'relu'
    dropout_rate    : float                             = 0.0
    learning_rate   : float                             = 1e-3
    epochs          : int                               = 200
    batch_size      : int                               = 512
    weight_decay    : float                             = 1e-4


@dataclass(frozen = True)
class HyperSVM:
    c               : float = 1.0
    learning_rate   : float = 1e-2
    epochs          : int   = 200
    batch_size      : int   = 512


@dataclass(frozen = True)
class HyperKAN:
    hidden          : Tuple[int, ...]   = (16,)
    grid_size       : int               = 8
    spline_order    : int               = 3
    grid_low        : float             = -3.0
    grid_high       : float             = 3.0
    learning_rate   : float             = 1e-3
    epochs          : int               = 200
    batch_size      : int               = 512
    weight_decay    : float             = 1e-4


@dataclass(frozen = True)
class HyperForest:
    n_trees             : int   = 64
    depth               : int   = 6
    n_bins              : int   = 32
    feature_fraction    : float = 0.7


@dataclass(frozen = True)
class HyperBoosting:
    n_rounds        : int   = 100
    depth           : int   = 4
    n_bins          : int   = 32
    learning_rate   : float = 0.1
    reg_lambda      : float = 1.0


type Hyper = HyperLogReg | HyperMLP | HyperSVM | HyperKAN | HyperForest | HyperBoosting


@dataclass(frozen = True)
class LinearState:
    w   : Array
    b   : Array


@dataclass(frozen = True)
class MlpState:
    hp      : HyperMLP
    classes : int
    params  : Params


@dataclass(frozen = True)
class KanState:
    hp      : HyperKAN
    classes : int
    params  : Params


@dataclass(frozen = True)
class ForestState:
    edges   : Array
    feats   : Array
    thr     : Array
    dist    : Array


@dataclass(frozen = True)
class BoostingState:
    edges           : Array
    feats           : Array
    thr             : Array
    value           : Array
    learning_rate   : float


type State = LinearState | MlpState | KanState | ForestState | BoostingState


@dataclass(frozen = True)
class Calc:
    @staticmethod
    def activation(name: str) -> Callable[[Array], Array]:
        match name:
            case 'relu': return nn.relu
            case 'tanh': return nn.tanh
            case 'gelu': return nn.gelu
            case _:      raise ValueError(f'неподдерживаемая активация: {name}')

    @staticmethod
    def softmax(z: Array) -> Array:
        return jx.nn.softmax(z, axis = -1)

    @staticmethod
    def one_hot(y: Array, classes: int) -> Array:
        return jx.nn.one_hot(y, classes)

    @staticmethod
    def wmean(values: Array, weights: Array) -> Array:
        return jp.sum(weights * values) / (jp.sum(weights) + 1e-8)


@dataclass(frozen = True)
class Optim:
    @staticmethod
    def descend(
            loss_fn         : Callable[[Params, Array, Array, Array, Key], Array],
            params          : Params,
            xs              : Array,
            ys              : Array,
            learning_rate   : float,
            epochs          : int,
            batch_size      : int,
            weight_decay    : float,
            key             : Key,
            val             : None | Tuple[Array, Array] = None,
            weights         : None | Array = None,
    ) -> Params:
        ws          = weights if weights is not None else jp.ones((xs.shape[0],), dtype = xs.dtype)
        opt         = ox.adamw(
            learning_rate   = learning_rate, weight_decay = weight_decay,
            mask            = lambda tree: jx.tree_util.tree_map(lambda leaf: leaf.ndim > 1, tree))
        opt_state   = opt.init(params)
        n           = xs.shape[0]
        d           = xs.shape[1]
        batch_sz    = min(batch_size, n)
        n_batches   = max(1, n // batch_sz)
        usable      = n_batches * batch_sz

        def batch_step(carry, batch):
            params, opt_state   = carry
            xb, yb, wb, bkey    = batch
            loss, grads         = jx.value_and_grad(loss_fn)(params, xb, yb, wb, bkey)
            updates, opt_state  = opt.update(grads, opt_state, params)
            return (ox.apply_updates(params, updates), opt_state), loss

        def run_epoch(params, opt_state, ekey):
            pkey, bkey  = jx.random.split(ekey)
            perm        = jx.random.permutation(pkey, n)[:usable]
            xb          = xs[perm].reshape(n_batches, batch_sz, d)
            yb          = ys[perm].reshape(n_batches, batch_sz)
            wb          = ws[perm].reshape(n_batches, batch_sz)
            bkeys       = jx.random.split(bkey, n_batches)
            (params, opt_state), losses = jx.lax.scan(batch_step, (params, opt_state), (xb, yb, wb, bkeys))
            return params, opt_state, jp.mean(losses)

        keys = jx.random.split(key, epochs)
        if val is None:
            def plain_epoch(carry, ekey):
                params, opt_state           = carry
                params, opt_state, mloss    = run_epoch(params, opt_state, ekey)
                return (params, opt_state), mloss
            (params, _), _ = jx.lax.scan(plain_epoch, (params, opt_state), keys)
            return params

        xs_val, ys_val = val
        ws_val         = jp.ones((xs_val.shape[0],), dtype = xs_val.dtype)
        def val_epoch(carry, ekey):
            params, opt_state, best, best_val   = carry
            params, opt_state, mloss            = run_epoch(params, opt_state, ekey)
            score                               = loss_fn(params, xs_val, ys_val, ws_val, ekey)
            improved                            = score < best_val
            best                                = jx.tree_util.tree_map(lambda b, p: jp.where(improved, p, b), best, params)
            return (params, opt_state, best, jp.where(improved, score, best_val)), mloss
        (_, _, best, _), _ = jx.lax.scan(val_epoch, (params, opt_state, params, jp.asarray(jp.inf)), keys)
        return best


@dataclass(frozen = True)
class Logistic:
    @staticmethod
    def fit(key: Key, xs: Array, ys: Array, classes: int, hp: HyperLogReg, val: None | Tuple[Array, Array] = None, weights: None | Array = None) -> LinearState:
        params  = {'w': jp.zeros((xs.shape[1], classes)), 'b': jp.zeros((classes,))}
        loss    = lambda p, xb, yb, wb, _bkey: Calc.wmean(Loss.cross_entropy(xb @ p['w'] + p['b'], yb, classes), wb)
        trained = Optim.descend(loss, params, xs, ys, hp.learning_rate, hp.epochs, hp.batch_size, hp.weight_decay, key, val = val, weights = weights)
        return LinearState(trained['w'], trained['b'])


@dataclass(frozen = True)
class Svm:
    @staticmethod
    def loss(params: Params, xb: Array, yb: Array, classes: int) -> Array:
        scores  = xb @ params['w'] + params['b']
        onehot  = Calc.one_hot(yb, classes)
        correct = jp.sum(scores * onehot, axis = 1, keepdims = True)
        margins = jp.maximum(0.0, 1.0 + scores - correct) * (1.0 - onehot)
        return jp.sum(margins, axis = 1)

    @staticmethod
    def fit(key: Key, xs: Array, ys: Array, classes: int, hp: HyperSVM, val: None | Tuple[Array, Array] = None, weights: None | Array = None) -> LinearState:
        params  = {'w': jp.zeros((xs.shape[1], classes)), 'b': jp.zeros((classes,))}
        loss    = lambda p, xb, yb, wb, _bkey: Calc.wmean(Svm.loss(p, xb, yb, classes), wb)
        trained = Optim.descend(loss, params, xs, ys, hp.learning_rate, hp.epochs, hp.batch_size, 1.0 / hp.c, key, val = val, weights = weights)
        return LinearState(trained['w'], trained['b'])


class MlpNet(nn.Module):
    hidden          : Tuple[int, ...]
    n_classes       : int
    activation      : str
    dropout_rate    : float

    @nn.compact
    def __call__(self, xs: Array, training: bool) -> Array:
        act     = Calc.activation(self.activation)
        drop    = lambda h: nn.Dropout(self.dropout_rate, deterministic = not training)(h) if self.dropout_rate > 0.0 else h
        body    = lambda h, size: drop(act(nn.Dense(size)(h)))
        return nn.Dense(self.n_classes)(reduce(body, self.hidden, xs))


@dataclass(frozen = True)
class Mlp:
    @staticmethod
    def module(hp: HyperMLP, classes: int) -> MlpNet:
        return MlpNet(hidden = hp.hidden, n_classes = classes, activation = hp.activation, dropout_rate = hp.dropout_rate)

    @staticmethod
    def fit(key: Key, xs: Array, ys: Array, classes: int, hp: HyperMLP, val: None | Tuple[Array, Array] = None, weights: None | Array = None) -> MlpState:
        model       = Mlp.module(hp, classes)
        ikey, tkey  = jx.random.split(key)
        params      = model.init({'params': ikey, 'dropout': ikey}, xs[:1], training = False)['params']
        loss        = lambda p, xb, yb, wb, bkey: Calc.wmean(Loss.cross_entropy(
            jp.asarray(model.apply({'params': p}, xb, training = True, rngs = {'dropout': bkey})), yb, classes), wb)
        trained     = Optim.descend(loss, params, xs, ys, hp.learning_rate, hp.epochs, hp.batch_size, hp.weight_decay, tkey, val = val, weights = weights)
        return MlpState(hp, classes, trained)

    @staticmethod
    def proba(state: MlpState, xs: Array) -> Array:
        model = Mlp.module(state.hp, state.classes)
        return Calc.softmax(jp.asarray(model.apply({'params': state.params}, xs, training = False)))


@dataclass(frozen = True)
class Spline:
    @staticmethod
    def grid(low: float, high: float, grid_size: int, order: int) -> Array:
        step = (high - low) / grid_size
        return low + (jp.arange(grid_size + 2 * order + 1) - order) * step

    @staticmethod
    def basis(xs: Array, grid: Array, order: int) -> Array:
        xe      = xs[..., None]
        degree0 = ((xe >= grid[:-1]) & (xe < grid[1:])).astype(xs.dtype)
        knots   = grid.shape[0]

        def lift(prev, p):
            left_den    = grid[p:knots - 1]             - grid[0:knots - p - 1]
            right_den   = grid[p + 1:knots]             - grid[1:knots - p]
            left        = (xe - grid[0:knots - p - 1])  / (left_den  + 1e-8) * prev[..., :-1]
            right       = (grid[p + 1:knots] - xe)      / (right_den + 1e-8) * prev[..., 1:]
            return left + right

        return reduce(lift, range(1, order + 1), degree0)


class KanLayer(nn.Module):
    out_dim         : int
    grid_size       : int
    spline_order    : int
    grid_low        : float
    grid_high       : float

    @nn.compact
    def __call__(self, xs: Array) -> Array:
        in_dim  = xs.shape[-1]
        grid    = Spline.grid(self.grid_low, self.grid_high, self.grid_size, self.spline_order)
        bases   = Spline.basis(xs, grid, self.spline_order)
        n_basis = self.grid_size + self.spline_order
        spline  = self.param('spline', initializers.normal(0.1),    (in_dim, self.out_dim, n_basis))
        base_w  = self.param('base',   initializers.lecun_normal(), (in_dim, self.out_dim))
        return nn.silu(xs) @ base_w + jp.einsum('nik,iok->no', bases, spline)


class KanNet(nn.Module):
    hidden          : Tuple[int, ...]
    n_classes       : int
    grid_size       : int
    spline_order    : int
    grid_low        : float
    grid_high       : float

    @nn.compact
    def __call__(self, xs: Array, training: bool) -> Array:
        layer = lambda h, size: KanLayer(size, self.grid_size, self.spline_order, self.grid_low, self.grid_high)(h)
        return reduce(layer, self.hidden + (self.n_classes,), xs)


@dataclass(frozen = True)
class Kan:
    @staticmethod
    def module(hp: HyperKAN, classes: int) -> KanNet:
        return KanNet(
            hidden          = hp.hidden, n_classes = classes, grid_size = hp.grid_size,
            spline_order    = hp.spline_order, grid_low = hp.grid_low, grid_high = hp.grid_high)

    @staticmethod
    def fit(key: Key, xs: Array, ys: Array, classes: int, hp: HyperKAN, val: None | Tuple[Array, Array] = None, weights: None | Array = None) -> KanState:
        model       = Kan.module(hp, classes)
        ikey, tkey  = jx.random.split(key)
        params      = model.init(ikey, xs[:1], training = False)['params']
        loss        = lambda p, xb, yb, wb, _bkey: Calc.wmean(Loss.cross_entropy(
            jp.asarray(model.apply({'params': p}, xb, training = True)), yb, classes), wb)
        trained     = Optim.descend(loss, params, xs, ys, hp.learning_rate, hp.epochs, hp.batch_size, hp.weight_decay, tkey, val = val, weights = weights)
        return KanState(hp, classes, trained)

    @staticmethod
    def proba(state: KanState, xs: Array) -> Array:
        model = Kan.module(state.hp, state.classes)
        return Calc.softmax(jp.asarray(model.apply({'params': state.params}, xs, training = False)))


@dataclass(frozen = True)
class Bins:
    @staticmethod
    def edges(xs: Array, n_bins: int) -> Array:
        return jp.quantile(xs, jp.arange(1, n_bins) / n_bins, axis = 0).T

    @staticmethod
    def digitize(xs: Array, edges: Array) -> Array:
        place = lambda col, edge: jp.searchsorted(edge, col, side = 'right')
        return jx.vmap(place, in_axes = (1, 0))(xs, edges).T


@dataclass(frozen = True)
class Tree:
    @staticmethod
    def histogram(binned: Array, leaf: Array, payload: Array, n_bins: int, n_leaves: int) -> Array:
        width   = payload.shape[1]
        one     = lambda col: jx.ops.segment_sum(
            payload, col * n_leaves + leaf, num_segments = n_bins * n_leaves).reshape(n_bins, n_leaves, width)
        return jx.vmap(one, in_axes = 1)(binned)

    @staticmethod
    def split_gini(hist: Array, feat_mask: Array) -> Tuple[Array, Array]:
        cum     = jp.cumsum(hist, axis = 1)
        total   = cum[:, -1]
        left    = cum[:, :-1]
        right   = total[:, None] - left
        ln      = jp.sum(left,  axis = -1)
        rn      = jp.sum(right, axis = -1)
        gini    = lambda counts, n: 1.0 - jp.sum((counts / (n[..., None] + 1e-8)) ** 2, axis = -1)
        impure  = jp.sum(ln * gini(left, ln) + rn * gini(right, rn), axis = -1)
        guarded = jp.where(feat_mask[:, None], impure, jp.inf)
        flat    = jp.argmin(guarded.reshape(-1))
        return flat // guarded.shape[1], flat % guarded.shape[1]

    @staticmethod
    def split_gain(hist: Array, feat_mask: Array, reg_lambda: float) -> Tuple[Array, Array]:
        cum     = jp.cumsum(hist, axis = 1)
        total   = cum[:, -1]
        left    = cum[:, :-1]
        right   = total[:, None] - left
        score   = lambda gh: gh[..., 0] ** 2 / (gh[..., 1] + reg_lambda)
        gain    = jp.sum(score(left) + score(right) - score(total)[:, None], axis = -1)
        guarded = jp.where(feat_mask[:, None], gain, -jp.inf)
        flat    = jp.argmax(guarded.reshape(-1))
        return flat // guarded.shape[1], flat % guarded.shape[1]

    @staticmethod
    def grow(binned: Array, payload: Array, feat_mask: Array, depth: int, n_bins: int, choose: Callable) -> Tuple[Array, Array, Array]:
        def level(carry, ell):
            leaf, feats, thr    = carry
            hist                = Tree.histogram(binned, leaf, payload, n_bins, 2 ** ell)
            best_f, best_k      = choose(hist, feat_mask)
            bit                 = (jp.take(binned, best_f, axis = 1) > best_k).astype(jp.int32)
            return leaf * 2 + bit, feats + (best_f,), thr + (best_k,)
        leaf0               = jp.zeros((binned.shape[0],), dtype = jp.int32)
        leaf, feats, thr    = reduce(level, range(depth), (leaf0, (), ()))
        totals              = jx.ops.segment_sum(payload, leaf, num_segments = 2 ** depth)
        return jp.stack(feats), jp.stack(thr), totals

    @staticmethod
    def leaf_index(binned: Array, feats: Array, thr: Array) -> Array:
        def descend(leaf, split):
            feat, thresh    = split
            bit             = (jp.take(binned, feat, axis = 1) > thresh).astype(jp.int32)
            return leaf * 2 + bit, None
        leaf, _ = jx.lax.scan(descend, jp.zeros((binned.shape[0],), dtype = jp.int32), (feats, thr))
        return leaf


@dataclass(frozen = True)
class ObliviousForest:
    @staticmethod
    def fit(key: Key, xs: Array, ys: Array, classes: int, hp: HyperForest, val: None | Tuple[Array, Array] = None, weights: None | Array = None) -> ForestState:
        edges   = Bins.edges(xs, hp.n_bins)
        binned  = Bins.digitize(xs, edges)
        onehot  = Calc.one_hot(ys, classes)
        n       = xs.shape[0]
        d       = xs.shape[1]
        ws      = weights if weights is not None else jp.ones((n,), dtype = xs.dtype)
        choose  = lambda hist, mask: Tree.split_gini(hist, mask)

        def one_tree(tkey):
            wkey, fkey      = jx.random.split(tkey)
            boot            = jx.ops.segment_sum(jp.ones((n,)), jx.random.randint(wkey, (n,), 0, n), num_segments = n)
            sample_w        = ws * boot
            drawn           = jx.random.bernoulli(fkey, hp.feature_fraction, (d,))
            mask            = jp.where(jp.any(drawn), drawn, jp.ones_like(drawn))
            feats, thr, tot = Tree.grow(binned, sample_w[:, None] * onehot, mask, hp.depth, hp.n_bins, choose)
            return feats, thr, tot / (jp.sum(tot, axis = -1, keepdims = True) + 1e-8)

        feats, thr, dist = jx.vmap(one_tree)(jx.random.split(key, hp.n_trees))
        return ForestState(edges, feats, thr, dist)

    @staticmethod
    def proba(state: ForestState, xs: Array) -> Array:
        binned  = Bins.digitize(xs, state.edges)
        vote    = lambda feats, thr, dist: dist[Tree.leaf_index(binned, feats, thr)]
        return jp.mean(jx.vmap(vote)(state.feats, state.thr, state.dist), axis = 0)


@dataclass(frozen = True)
class ObliviousBoosting:
    @staticmethod
    def fit(key: Key, xs: Array, ys: Array, classes: int, hp: HyperBoosting, val: None | Tuple[Array, Array] = None, weights: None | Array = None) -> BoostingState:
        edges   = Bins.edges(xs, hp.n_bins)
        binned  = Bins.digitize(xs, edges)
        onehot  = Calc.one_hot(ys, classes)
        ws      = (weights if weights is not None else jp.ones((xs.shape[0],), dtype = xs.dtype))[:, None]
        mask    = jp.ones((xs.shape[1],), dtype = bool)
        choose  = lambda hist, m: Tree.split_gain(hist, m, hp.reg_lambda)
        predict = lambda feats, thr, value: jx.vmap(lambda f, t, v: v[Tree.leaf_index(binned, f, t)])(feats, thr, value)

        def boost(scores, _rkey):
            proba               = Calc.softmax(scores)
            grad                = (proba - onehot) * ws
            hess                = proba * (1.0 - proba) * ws
            fit_one             = lambda g, h: Tree.grow(binned, jp.stack((g, h), axis = 1), mask, hp.depth, hp.n_bins, choose)
            feats, thr, totals  = jx.vmap(fit_one, in_axes = 1)(grad, hess)
            value               = -totals[..., 0] / (totals[..., 1] + hp.reg_lambda)
            return scores + hp.learning_rate * predict(feats, thr, value).T, (feats, thr, value)

        scores, rounds  = jx.lax.scan(boost, jp.zeros((xs.shape[0], classes)), jx.random.split(key, hp.n_rounds))
        feats, thr, values = rounds

        return BoostingState(edges, feats, thr, values, hp.learning_rate)

    @staticmethod
    def proba(state: BoostingState, xs: Array) -> Array:
        binned  = Bins.digitize(xs, state.edges)
        classes = state.value.shape[1]
        predict = lambda feats, thr, value: jx.vmap(lambda f, t, v: v[Tree.leaf_index(binned, f, t)])(feats, thr, value)

        def replay(scores, rnd):
            feats, thr, value = rnd
            return scores + state.learning_rate * predict(feats, thr, value).T, None

        scores, _ = jx.lax.scan(replay, jp.zeros((xs.shape[0], classes)), (state.feats, state.thr, state.value))
        
        return Calc.softmax(scores)


@dataclass(frozen = True)
class Learner:
    @staticmethod
    def fit(key: Key, xs: Array, ys: Array, classes: int, hp: Hyper, val: None | Tuple[Array, Array] = None, weights: None | Array = None) -> State:
        match hp:
            case HyperLogReg():   return Logistic.fit(key, xs, ys, classes, hp, val, weights)
            case HyperSVM():      return Svm.fit(key, xs, ys, classes, hp, val, weights)
            case HyperMLP():      return Mlp.fit(key, xs, ys, classes, hp, val, weights)
            case HyperKAN():      return Kan.fit(key, xs, ys, classes, hp, val, weights)
            case HyperForest():   return ObliviousForest.fit(key, xs, ys, classes, hp, val, weights)
            case HyperBoosting(): return ObliviousBoosting.fit(key, xs, ys, classes, hp, val, weights)
            case _:               raise ValueError(f'неизвестный тип гиперпараметров: {hp!r}')

    @staticmethod
    def proba(state: State, xs: Array) -> Array:
        match state:
            case LinearState():     return Calc.softmax(xs @ state.w + state.b)
            case MlpState():        return Mlp.proba(state, xs)
            case KanState():        return Kan.proba(state, xs)
            case ForestState():     return ObliviousForest.proba(state, xs)
            case BoostingState():   return ObliviousBoosting.proba(state, xs)
            case _:                 raise ValueError(f'неизвестное состояние модели: {state}')