from typing                 import TypeVar, Generic, Iterable, Callable, Literal, Tuple, Dict, Any, overload, cast
from abc                    import abstractmethod
from dataclasses            import dataclass
from functools              import reduce, wraps
from itertools              import chain

from codebase_node_abstract import Functor, Fix, Algebra, cata


A = TypeVar('A')
B = TypeVar('B')

CompositionOps  = Literal['>>', '<<']
ParallelOp      = Literal['|']
Operator        = CompositionOps | ParallelOp

@dataclass(frozen = True, slots = True)
class NormalizerConfig:
    precedence      : Dict[Operator, int]
    associativity   : Dict[CompositionOps, Literal['L', 'R']]

    @classmethod
    def default(cls) -> 'NormalizerConfig':
        return cls(
            precedence      = {'|': 1, '>>': 2, '<<': 2},
            associativity   = {'>>': 'L', '<<': 'L'}
        )


class CompositionF(Generic[A], Functor[A]):
    @abstractmethod
    def map(self, f: Callable[[A], B]) -> 'CompositionF[B]': ...

@dataclass(frozen = True, slots = True)
class FuncF(CompositionF[A]):
    func: Callable
    
    def map(self, f: Callable[[A], B]) -> 'FuncF[B]':
        return FuncF(func = self.func)

@dataclass(frozen = True, slots = True)
class BinF(CompositionF[A]):
    op: CompositionOps

    l: A
    r: A

    def map(self, f: Callable[[A], B]) -> 'BinF[B]':
        return BinF(op = self.op, l = f(self.l), r = f(self.r))

@dataclass(frozen = True, slots = True)
class ParF(CompositionF[A]):
    branches: Tuple[A, ...]

    def map(self, f: Callable[[A], B]) -> 'ParF[B]':
        return ParF(branches = tuple(map(f, self.branches)))


def evaluate_algebra(node_f: CompositionF[Callable]) -> Callable:
    match node_f:
        case FuncF(func = f):               return lambda *a, **k: f(*a, **k)
        case BinF(op = '>>', l = f, r = g): return lambda *a, **k: g(f(*a, **k))
        case BinF(op = '<<', l = f, r = g): return lambda *a, **k: f(g(*a, **k))
        case ParF(branches = bs):           return lambda *a, **k: tuple(map(lambda b: b(*a, **k), bs))

        case _: raise TypeError(f'неизвестный узел: {node_f}')

def compile_algebra(node_f: CompositionF[Callable]) -> Callable:
    match node_f:
        case FuncF(func = f):               return lambda *a, **k: f(*a, **k)
        case BinF(op = '>>', l = f, r = g): return lambda *a, **k: g(f(*a, **k))
        case BinF(op = '<<', l = f, r = g): return lambda *a, **k: f(g(*a, **k))
        case ParF(branches = bs):           return lambda *a, **k: tuple(map(lambda b: b(*a, **k), bs))

        case _: raise TypeError(f'неизвестный узел: {node_f}')

def show_algebra(node_f: CompositionF[str]) -> str:
    match node_f:
        case FuncF(func = f):               return getattr(f, '__name__', str(f))
        case BinF(op = op, l = l, r = r):   return f'({l} {op} {r})'
        case ParF(branches = bs):
            if len(bs) == 1: return bs[0]

            return '(' + ' | '.join(bs) + ')'

        case _: raise TypeError(f'неизвестный узел: {node_f}')

def normalize_algebra(config: NormalizerConfig) -> Callable[[CompositionF[Fix]], Fix]:
    def _normalize_binary(
        op      : CompositionOps,
        l       : Fix,
        r       : Fix,
    ) -> Fix:
        prec    = config.precedence[op]
        assoc   = config.associativity[op]

        def __rotate_r(node: Fix) -> None | Fix:
            match node.o():
                case BinF(op = child_op, l = child_l, r = child_r):
                    child_prec = config.precedence[child_op]

                    if child_prec < prec or (child_prec == prec and assoc == 'L'):
                        new_l = Fix.i(BinF(op = op, l = l, r = child_l))

                        return _normalize_binary(child_op, new_l, child_r) #config
                case _: pass

            return None

        def __rotate_l(node: Fix) -> None | Fix:
            match node.o():
                case BinF(op = child_op, l = child_l, r = child_r):
                    child_prec = config.precedence[child_op]

                    if child_prec < prec or (child_prec == prec and assoc == 'R'):
                        new_r = Fix.i(BinF(op = op, l = child_r, r = r))

                        return _normalize_binary(child_op, child_l, new_r) #config
                case _: pass

            return None

        rotated = __rotate_r(r) or __rotate_l(l)
        if rotated is not None: return rotated

        return Fix.i(BinF(op = op, l = l, r = r))

    def _algebra(node_f: CompositionF[Fix]) -> Fix:
        match node_f:
            case BinF(op = op, l = l, r = r):
                return _normalize_binary(op, l, r)
            
            case ParF(branches = bs):
                flattened = tuple(chain.from_iterable(map(
                    lambda b: cast(ParF, b.o()).branches if isinstance(b.o(), ParF) else (b,),
                    bs)))

                return Fix.i(ParF(branches = flattened))
            
            case FuncF():
                return Fix.i(node_f)

            case _: raise TypeError(f'неизвестный узел: {node_f}')

    return _algebra


class Composition:
    def __init__(
        self,
        value               : Callable | Fix | Iterable[Callable],
        normalize_config    : NormalizerConfig
    ) -> None:
        match value:
            case Fix():                 ast = value
            case _ if callable(value):  ast = Fix.i(FuncF(func=value))
            case _:
                funcs = tuple(cast(Iterable[Callable], value))
                if not funcs: raise ValueError('композиция не может быть пустой')

                ast = Fix.i(ParF(branches = tuple(map(lambda f: Fix.i(FuncF(func = f)), funcs))))

        if normalize_config is not None:
            ast = cata(cast(Algebra[Fix], normalize_algebra(normalize_config)), ast)

        self._ast               : Fix               = ast
        self._normalize_config  : NormalizerConfig  = normalize_config

        self._compiled_func     : None | Callable   = None
        self._cached_evaluator  : None | Callable   = None
    
    def __repr__(self) -> str:
        compiled = 'скомпилирована' if self._compiled_func else 'интерпретируется'
        
        return f'<Composition {compiled} ast={self.show()}>'

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._compiled_func is not None: return self._compiled_func(*args, **kwargs)
        if self._cached_evaluator is None:
            self._cached_evaluator = cata(cast(Algebra[Fix], evaluate_algebra), self._ast)
        assert self._cached_evaluator is not None

        return self._cached_evaluator(*args, **kwargs)

    def compile(self) -> 'Composition':
        if self._compiled_func is None:
            self._compiled_func = cata(cast(Algebra[Fix], compile_algebra), self._ast)

        return self

    def show(self) -> str:
        return cata(cast(Algebra[Fix], show_algebra), self._ast)

    def normalize(self, config: NormalizerConfig = NormalizerConfig.default()) -> 'Composition':
        normalized_ast = cata(cast(Algebra[Fix], normalize_algebra(config)), self._ast)

        return Composition(normalized_ast, normalize_config = config)

    def _binary(self, op: CompositionOps, other: Callable | 'Composition') -> 'Composition':
        other_ast = other._ast if isinstance(other, Composition) else Fix.i(FuncF(func = other))

        return Composition(Fix.i(BinF(op=op, l=self._ast, r=other_ast)), self._normalize_config)

    def _par(self, other_ast: Fix) -> 'Composition':
        lb = cast(ParF, self._ast.o()).branches if isinstance(self._ast.o(), ParF) else (self._ast,)
        rb = cast(ParF, other_ast.o()).branches if isinstance(other_ast.o(), ParF) else (other_ast,)

        return Composition(Fix.i(ParF(branches = lb + rb)), self._normalize_config)

    def __rshift__( self, other: Callable) -> 'Composition':    return self._binary('>>', other)
    def __rrshift__(self, other: Callable) -> 'Composition':    return Composition(other, self._normalize_config)._binary('>>', self)
    def __lshift__( self, other: Callable) -> 'Composition':    return self._binary('<<', other)
    def __rlshift__(self, other: Callable) -> 'Composition':    return Composition(other, self._normalize_config)._binary('<<', self)
    def __or__(     self, other: Callable) -> 'Composition':    return self._par(
                                                                                other._ast
                                                                                if isinstance(other, Composition) else
                                                                                Fix.i(FuncF(func = other)))
    def __ror__(    self, other: Callable) -> 'Composition':    return Composition(other, self._normalize_config) | self


@overload
def composable(func: Callable) -> Composition: ...
@overload
def composable(*, unpack_params: Literal[None, '*', '**'] = None) -> Callable[[Callable], Composition]: ...

def composable(
    func: None | Callable = None,
    *,
    n_cfg: NormalizerConfig = NormalizerConfig.default(),
    unpack_params: None | Literal['*', '**'] = None
) -> Composition | Callable[[Callable], Composition]:
    def _decorator(f: Callable) -> Composition:
        match unpack_params:
            case '*':   wrapped = wraps(f)(lambda args: f(*args))
            case '**':  wrapped = wraps(f)(lambda kwargs: f(**kwargs))
            case _:     wrapped = f
        
        return Composition(Fix.i(FuncF(func = wrapped)), n_cfg)
    
    if func is None: return _decorator

    return _decorator(func)

@composable
def identity(x: A) -> A: return x

def compose(*funcs: Callable | Composition) -> Composition:
    return reduce(lambda acc, f: acc << f, funcs, identity)

def pipe(*funcs: Callable | Composition) -> Composition:
    return reduce(lambda acc, f: acc >> f, funcs, identity)

def parallel(*funcs: Callable | Composition) -> Composition:
    if not funcs: return identity
    
    return reduce(lambda acc, f: acc | f, cast(Iterable[Composition], funcs))

# todo: избавиться от cast и Any
# todo: добавить инфиксный оператор декорирования: (h >> dec_outer @ dec_inner @ f >> g)(x) == g(dec_outer(dec_inner(f))(h(x)))
# todo: добавить возможность монадической композиции функций (в моноидальных контекстах, например, композиция нескольких функции,
# обернутых в таймеры возвращает результат применения композиции исходных функций к входному аргументу и суммарное время исполнения)