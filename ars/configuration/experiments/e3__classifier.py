from    typing                                  import Tuple
from    dataclasses                             import dataclass

from    ars.models.m3__classifier.architecture  import (
    Kind, HyperLogReg, HyperMLP, HyperSVM, HyperKAN, HyperForest, HyperBoosting, Hyper)


@dataclass(frozen = True)
class Experiment:
    name        : str               = ''
    bases       : Tuple[Kind, ...]  = ('logreg', 'mlp', 'svm', 'kan', 'forest', 'boosting')
    meta_solver : Kind              = 'logreg'
    n_folds     : int               = 5
    seed        : int               = 12345

    logreg      : HyperLogReg   = HyperLogReg()
    mlp         : HyperMLP      = HyperMLP()
    svm         : HyperSVM      = HyperSVM()
    kan         : HyperKAN      = HyperKAN()
    forest      : HyperForest   = HyperForest()
    boosting    : HyperBoosting = HyperBoosting()

    def hyper(self, kind: Kind) -> Hyper:
        match kind:
            case 'logreg':   return self.logreg
            case 'mlp':      return self.mlp
            case 'svm':      return self.svm
            case 'kan':      return self.kan
            case 'forest':   return self.forest
            case 'boosting': return self.boosting
            case _:          raise ValueError(f'неизвестная базовая модель: {kind}')


@dataclass(frozen = True)
class Baseline(Experiment):
    name    : str   = 'baseline'


@dataclass(frozen = True)
class TreesEnsemble(Experiment):
    name        : str               = 'trees_ensemble'
    bases       : Tuple[Kind, ...]  = ('forest', 'boosting', 'logreg')
    meta_solver : Kind              = 'boosting'
    forest      : HyperForest       = HyperForest(n_trees = 64, depth = 6)
    boosting    : HyperBoosting     = HyperBoosting(n_rounds = 200, depth = 6, learning_rate = 0.05)


@dataclass(frozen = True)
class NeuralStack(Experiment):
    name        : str               = 'neural_stack'
    bases       : Tuple[Kind, ...]  = ('mlp', 'kan', 'logreg', 'svm')
    meta_solver : Kind              = 'mlp'
    mlp         : HyperMLP          = HyperMLP(hidden = (256, 128), epochs = 300, dropout_rate = 0.1)
    kan         : HyperKAN          = HyperKAN(hidden = (32,), grid_size = 12, epochs = 300)


@dataclass(frozen = True)
class DeepBoost(Experiment):
    name        : str               = 'deep_boost'
    bases       : Tuple[Kind, ...]  = ('boosting', 'forest', 'mlp', 'logreg')
    meta_solver : Kind              = 'logreg'
    boosting    : HyperBoosting     = HyperBoosting(n_rounds = 300, depth = 6, learning_rate = 0.03, reg_lambda = 2.0)
    forest      : HyperForest       = HyperForest(n_trees = 64, depth = 6)


EXPERIMENTS : Tuple[type[Experiment], ...]  = (
    #Baseline,
    TreesEnsemble,
    #NeuralStack,
    #DeepBoost,
)


def show_experiment(e: type[Experiment]) -> None:
    x = e()
    print(x.name)
    print(f'базовые: {x.bases}')
    print(f'мета-решатель: {x.meta_solver} | фолдов: {x.n_folds}')
    print()