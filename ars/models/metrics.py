from    typing      import Literal, Tuple
from    dataclasses import dataclass
from    functools   import partial

import  jax         as jx
import  jax.numpy   as jp


type Array      = jp.ndarray
type Numeric    = Array | float | int


@dataclass(frozen = True)
class MetricSet:
    accuracy    : Numeric
    precision   : Numeric
    recall      : Numeric
    sensitivity : Numeric
    specificity : Numeric
    f1          : Numeric
    youden      : Numeric

    @staticmethod
    def get_names() -> Tuple[str, ...]:
        return tuple(MetricSet.__dataclass_fields__)


jx.tree_util.register_pytree_node(
    MetricSet,
    lambda metric_set: (tuple(map(partial(getattr, metric_set), MetricSet.get_names())), None),
    lambda _aux, children: MetricSet(*children))


type MetricName = Literal['accuracy', 'precision', 'recall', 'sensitivity', 'specificity', 'f1', 'youden']
type LossKind   = Literal['mse', 'huber']


@dataclass(frozen = True)
class Operators:
    @staticmethod
    def to_float(metric_set: MetricSet) -> MetricSet:
        return jx.tree_util.tree_map(lambda value: float(value), metric_set)

    @staticmethod
    def named(metric_set: MetricSet, name: MetricName) -> Numeric:
        return getattr(metric_set, name)


@dataclass(frozen = True)
class Confusion:
    @staticmethod
    def accuracy(tp: Numeric, tn: Numeric, fp: Numeric, fn: Numeric, eps: Numeric) -> Numeric:
        return (tp + tn) / (tp + tn + fp + fn + eps)

    @staticmethod
    def precision(tp: Numeric, fp: Numeric, eps: Numeric) -> Numeric:
        return tp / (tp + fp + eps)

    @staticmethod
    def recall(tp: Numeric, fn: Numeric, eps: Numeric) -> Numeric:
        return tp / (tp + fn + eps)

    @staticmethod
    def specificity(tn: Numeric, fp: Numeric, eps: Numeric) -> Numeric:
        return tn / (tn + fp + eps)

    @staticmethod
    def f1(tp: Numeric, fp: Numeric, fn: Numeric, eps: Numeric) -> Numeric:
        prec    = Confusion.precision(tp, fp, eps)
        rec     = Confusion.recall(tp, fn, eps)

        return 2.0 * prec * rec / (prec + rec + eps)

    @staticmethod
    def youden(tp: Numeric, tn: Numeric, fp: Numeric, fn: Numeric, eps: Numeric) -> Numeric:
        return Confusion.recall(tp, fn, eps) + Confusion.specificity(tn, fp, eps) - 1.0

    @staticmethod
    def metrics(tp: Numeric, tn: Numeric, fp: Numeric, fn: Numeric, eps: Numeric) -> MetricSet:
        return MetricSet(
            accuracy    = Confusion.accuracy    (tp, tn, fp, fn, eps),
            precision   = Confusion.precision   (tp, fp, eps),
            recall      = Confusion.recall      (tp, fn, eps),
            sensitivity = Confusion.recall      (tp, fn, eps),
            specificity = Confusion.specificity (tn, fp, eps),
            f1          = Confusion.f1          (tp, fp, fn, eps),
            youden      = Confusion.youden      (tp, tn, fp, fn, eps))


@dataclass(frozen = True)
class Binary:
    @staticmethod
    @jx.jit
    def measures(labels_int: Array, preds_int: Array, eps: Array) -> MetricSet:
        tp  = jp.sum((preds_int == 1) & (labels_int == 1)).astype(jp.float32)
        tn  = jp.sum((preds_int == 0) & (labels_int == 0)).astype(jp.float32)
        fp  = jp.sum((preds_int == 1) & (labels_int == 0)).astype(jp.float32)
        fn  = jp.sum((preds_int == 0) & (labels_int == 1)).astype(jp.float32)

        return Confusion.metrics(tp, tn, fp, fn, eps)

    @staticmethod
    def confusion(labels: Numeric, predictions: Numeric) -> Tuple[Array, Array, Array, Array]:
        labels_int  = jp.asarray(labels,      dtype = jp.int32)
        preds_int   = jp.asarray(predictions, dtype = jp.int32)

        return (jp.sum((preds_int == 1) & (labels_int == 1)).astype(jp.float32),
                jp.sum((preds_int == 0) & (labels_int == 0)).astype(jp.float32),
                jp.sum((preds_int == 1) & (labels_int == 0)).astype(jp.float32),
                jp.sum((preds_int == 0) & (labels_int == 1)).astype(jp.float32))

    @staticmethod
    def evaluate(labels: Numeric, predictions: Numeric, eps: float) -> MetricSet:
        return Operators.to_float(Binary.measures(
            jp.asarray(labels,      dtype = jp.int32),
            jp.asarray(predictions, dtype = jp.int32),
            jp.asarray(eps,         dtype = jp.float32)))


@dataclass(frozen = True)
class Multi:
    @staticmethod
    def confusion_matrix(labels: Array, predictions: Array, classes: int) -> Array:
        return jx.ops.segment_sum(
            jp.ones(labels.shape[0]),
            labels * classes + predictions,
            num_segments = classes * classes).reshape(classes, classes)

    @staticmethod
    def per_class(matrix: Array) -> Tuple[Array, Array, Array, Array]:
        diag    = jp.diag(matrix)
        actual  = jp.sum(matrix, axis = 1)
        predict = jp.sum(matrix, axis = 0)
        total   = jp.sum(matrix)

        return diag, total - predict - actual + diag, predict - diag, actual - diag

    @staticmethod
    def macro(matrix: Array, eps: Numeric) -> MetricSet:
        tp, tn, fp, fn  = Multi.per_class(matrix)
        per             = Confusion.metrics(tp, tn, fp, fn, eps)

        return MetricSet(
            accuracy    = jp.sum(jp.diag(matrix)) / (jp.sum(matrix) + eps),
            precision   = jp.mean(per.precision),
            recall      = jp.mean(per.recall),
            sensitivity = jp.mean(per.sensitivity),
            specificity = jp.mean(per.specificity),
            f1          = jp.mean(per.f1),
            youden      = jp.mean(per.youden))

    @staticmethod
    def micro(matrix: Array, eps: Numeric) -> MetricSet:
        tp, tn, fp, fn  = Multi.per_class(matrix)
        pooled          = Confusion.metrics(jp.sum(tp), jp.sum(tn), jp.sum(fp), jp.sum(fn), eps)

        return MetricSet(
            accuracy    = jp.sum(jp.diag(matrix)) / (jp.sum(matrix) + eps),
            precision   = pooled.precision,
            recall      = pooled.recall,
            sensitivity = pooled.sensitivity,
            specificity = pooled.specificity,
            f1          = pooled.f1,
            youden      = pooled.youden)

    @staticmethod
    @partial(jx.jit, static_argnums = (2,))
    def summary_proba_macro(labels: Array, proba: Array, classes: int, eps: float) -> MetricSet:
        predictions = jp.argmax(proba, axis = -1)

        return Multi.macro(Multi.confusion_matrix(labels, predictions, classes), eps)

    @staticmethod
    @partial(jx.jit, static_argnums = (2,))
    def summary_proba_micro(labels: Array, proba: Array, classes: int, eps: float) -> MetricSet:
        predictions = jp.argmax(proba, axis = -1)

        return Multi.micro(Multi.confusion_matrix(labels, predictions, classes), eps)

    @staticmethod
    def evaluate_macro(labels: Array, proba: Array, classes: int, eps: float) -> MetricSet:
        return Operators.to_float(Multi.summary_proba_macro(labels, proba, classes, eps))

    @staticmethod
    def evaluate_micro(labels: Array, proba: Array, classes: int, eps: float) -> MetricSet:
        return Operators.to_float(Multi.summary_proba_micro(labels, proba, classes, eps))


@dataclass(frozen = True)
class Loss:
    @staticmethod
    def pointwise(recon: Array, target: Array, kind: LossKind, delta: Numeric) -> Array:
        abs_error = jp.abs(recon - target)
        match kind:
            case 'mse':     return abs_error ** 2
            case 'huber':
                quadratic   = 0.5 * abs_error ** 2
                linear      = delta * (abs_error - 0.5 * delta)

                return jp.where(abs_error <= delta, quadratic, linear)
            case _:         raise NotImplementedError(f'неподдерживаемая функция потерь: {kind}')

    @staticmethod
    def mean(recon: Array, target: Array, kind: LossKind, delta: Numeric) -> Array:
        return jp.mean(Loss.pointwise(recon, target, kind, delta))

    @staticmethod
    def masked(recon: Array, target: Array, mask: Array, kind: LossKind, delta: Numeric, eps: Numeric) -> Array:
        # F-23: normalize per element (valid timesteps × feature dim), not per
        # timestep — otherwise branch losses scale with feature dimension and
        # EPI/SEM/Combined error magnitudes are incomparable.
        mask_e      = mask[..., None] if mask.ndim == 2 else mask
        pointwise   = Loss.pointwise(recon, target, kind, delta)
        n_elements  = jp.sum(mask_e) * recon.shape[-1]

        return jp.sum(pointwise * mask_e) / (n_elements + eps)

    @staticmethod
    def cross_entropy(logits: Array, labels: Array, classes: int) -> Array:
        return -jp.sum(jx.nn.log_softmax(logits, axis = -1) * jx.nn.one_hot(labels, classes), axis = -1)