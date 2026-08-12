from    typing                                          import Tuple
from    dataclasses                                     import dataclass
from    functools                                       import reduce
from    itertools                                       import starmap

import  jax                                             as jx
import  jax.numpy                                       as jp

from    ars.models.metrics                              import MetricSet, Multi
from    ars.models.m3__classifier.architecture          import Array, Key, State, Kind, Learner, Hyper
from    ars.configuration.experiments.e3__classifier    import Experiment


@dataclass(frozen = True)
class StackState:
    base_kinds  : Tuple[Kind, ...]
    base_states : Tuple[State, ...]
    meta_state  : State
    classes     : int


@dataclass(frozen = True)
class Folds:
    @staticmethod
    def assign(key: Key, n: int, k: int) -> Array:
        return jp.argsort(jp.argsort(jx.random.uniform(key, (n,)))) % k


@dataclass(frozen = True)
class Stack:
    @staticmethod
    def oof(key: Key, xs: Array, ys: Array, classes: int, hp: Hyper, folds: Array, k: int, val: None | Tuple[Array, Array] = None) -> Array:
        def for_fold(acc, fold):
            fkey    = jx.random.fold_in(key, fold)
            keep    = (folds != fold).astype(xs.dtype)
            state   = Learner.fit(fkey, xs, ys, classes, hp, val, keep)
            return jp.asarray(jp.where((folds == fold)[:, None], Learner.proba(state, xs), acc))

        return reduce(for_fold, range(k), jp.zeros((xs.shape[0], classes)))

    @staticmethod
    def fit(key: Key, xs: Array, ys: Array, classes: int, exp: Experiment, val: None | Tuple[Array, Array] = None) -> StackState:
        fkey, bkey, mkey    = jx.random.split(key, 3)
        folds               = Folds.assign(fkey, xs.shape[0], exp.n_folds)
        meta_features       = jp.concatenate(tuple(starmap(
            lambda i, kind: Stack.oof(jx.random.fold_in(bkey, i), xs, ys, classes, exp.hyper(kind), folds, exp.n_folds, val),
            enumerate(exp.bases))), axis = 1)
        base_states         = tuple(starmap(
            lambda i, kind: Learner.fit(jx.random.fold_in(bkey, i), xs, ys, classes, exp.hyper(kind), val),
            enumerate(exp.bases)))
        meta_state          = Learner.fit(mkey, meta_features, ys, classes, exp.hyper(exp.meta_solver))

        return StackState(exp.bases, base_states, meta_state, classes)

    @staticmethod
    def proba(stack: StackState, xs: Array) -> Array:
        base = tuple(map(lambda state: Learner.proba(state, xs), stack.base_states))

        return Learner.proba(stack.meta_state, jp.concatenate(base, axis = 1))

    @staticmethod
    def evaluate(stack: StackState, xs: Array, ys: Array, eps: float) -> MetricSet:
        return Multi.evaluate_macro(ys, Stack.proba(stack, xs), stack.classes, eps)