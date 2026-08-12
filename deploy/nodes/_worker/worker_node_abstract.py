from typing         import TypeVar, Generic, Protocol, Callable, Tuple, runtime_checkable
from abc            import abstractmethod
from dataclasses    import dataclass


A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
D = TypeVar('D')

T = TypeVar('T', covariant = True)
U = TypeVar('U')

P = TypeVar('P', covariant = True)
Q = TypeVar('Q', covariant = True)
R = TypeVar('R')
S = TypeVar('S')

# ---------- категории и алгебры ---------- #

@runtime_checkable
class Functor(Generic[T], Protocol):
    '''
    Функтор. Тип с одним ковариантным параметром и методом map,
    который применяет функцию к содержимому, не меняя структуру.
    '''
    
    @abstractmethod
    def map(self, f: Callable[[T], U]) -> 'Functor[U]': ...

@runtime_checkable
class Bifunctor(Generic[P, Q], Protocol):
    '''
    Бифунктор. Тип с двумя ковариантными параметрами и методом bimap,
    который применяет две функции независимо к каждому параметру.
    '''
    
    @abstractmethod
    def bimap(self, f: Callable[[P], R], g: Callable[[Q], S]) -> 'Bifunctor[R, S]': ...

Algebra     = Callable[[Functor[A]], A]             # Алгебра: функция, которая сворачивает один слой функтора в значение.
Coalgebra   = Callable[[A], Functor[A]]             # Коалгебра: функция, которая разворачивает значение в один слой функтора.
ZygoAlgebra = Callable[[Functor[Tuple[A, B]]], A]   # Алгебра для зигоморфизма: использует функтор, содержащий пары (основной результат, вспомогательный результат).


class Fix:
    '''
    Тип неподвижной точки (для функтора). Рекурсивная структура данных,
    построенная путём "зацикливания" функтора.
    '''
    
    def __init__(self, unfix: 'Functor[Fix]'):
        self.unfix = unfix

    @classmethod
    def i(cls, f: 'Functor[Fix]') -> 'Fix': return cls(f)
    def o(self) -> 'Functor[Fix]':          return self.unfix

RAlgebra    = Callable[[Functor[Tuple[Fix, A]]], A] # R-алгебра для параморфизма: использует функтор, содержащий пары (исходное поддерево Fix, результат A).
RCoalgebra  = Callable[[A], Functor[Fix | A]]       # R-коалгебра для апоморфизма: из значения порождает слой функтора, содержащий либо готовое Fix‑поддерево, либо новое значение для продолжения развёртки.


class Free(Generic[A]):
    '''
    Свободная монада со структорой дерева, в котором листья содержат значения типа A, а узлы —
    функтор, оборачивающий дальнейшие Free‑поддеревья.
    '''
    
    def __init__(self, value: A | 'Functor[Free[A]]'):
        self.value = value
    
    @staticmethod
    def pure(a: A) -> 'Free[A]':
        ''' создаёт чистое значение (лист) в свободной монаде '''

        return Free(a)

    @staticmethod
    def roll(f: 'Functor[Free[A]]') -> 'Free[A]':
        ''' создаёт узел, содержащий функтор с поддеревьями '''

        return Free(f)

class Cofree(Generic[A]):
    '''
    Свободная комонада: бесконечное дерево, в каждом узле которого хранится
    значение типа A и функтор с поддеревьями.
    '''
    
    def __init__(self, head: A, tail: 'Functor[Cofree[A]]'):
        self.head, self.tail = head, tail

CVAlgebra   = Callable[[Functor[Cofree[A]]], A]     # CV-алгебра для гистоморфизма: использует функтор, содержащий Cofree с историей всех предыдущих вычислений.
CVCoalgebra = Callable[[A], Functor[Free[A]]]       # CV-коалгебра для футуроморфизма: из значения порождает слой функтора, содержащий Free‑структуры, позволяющие генерировать несколько слоёв за один шаг.

# ---------- рекурсивные схемы ---------- #

def cata(alg: Algebra, fix: Fix) -> A:
    '''
    Катаморфизм (свёртка). Рекурсивно обходит Fix, заменяя конструкторы на
    вычисления, двигаясь от листьев к корню.
    '''

    return alg(fix.o().map(lambda sub: cata(alg, sub)))


def ana(coalg: Coalgebra, seed: A) -> Fix:
    '''
    Анаморфизм (развёртка). Строит Fix из начального значения, порождая слои
    функтора один за другим.
    '''

    return Fix.i(coalg(seed).map(lambda new_seed: ana(coalg, new_seed)))


def hylo(alg: Algebra, coalg: Coalgebra, seed: A) -> B:
    '''
    Хиломорфизм. Композиция анаморфизма и катаморфизма: строит промежуточную "виртуальную"
    структуру и тут же сворачивает её, не сохраняя в памяти целиком.
    '''

    return alg(coalg(seed).map(lambda sub: hylo(alg, coalg, sub)))


def para(alg: RAlgebra, fix: Fix) -> A:
    '''
    Параморфизм. Обобщает катаморфизм, предоставляя алгебре доступ не только
    к результатам рекурсивной обработки поддеревьев, но и к самим исходным поддеревьям.
    '''

    return alg(fix.o().map(lambda sub: (sub, para(alg, sub))))


def apo(coalg: RCoalgebra, seed: A) -> Fix:
    '''
    Апоморфизм. Обобщает анаморфизм, позволяя коалгебре досрочно вернуть
    готовое Fix‑поддерево, завершив развёртку на этом шаге.
    '''

    def _go(x: Fix | A) -> Fix:
        return x if isinstance(x, Fix) else apo(coalg, x)

    return Fix.i(coalg(seed).map(_go))


def histo(alg: CVAlgebra, fix: Fix) -> A:
    '''
    Гистоморфизм. Катаморфизм с доступом к истории всех предыдущих вычислений.
    Алгебра получает функтор, содержащий Cofree, где сохранены результаты для всех поддеревьев.
    '''

    def _build_cv(_fix: Fix) -> Cofree[A]:
        tail = _fix.o().map(_build_cv)

        return Cofree(alg(tail), tail)
    
    return _build_cv(fix).head


def futu(coalg: CVCoalgebra, seed: A) -> Fix:
    '''
    Футуроморфизм. Анаморфизм, позволяющий коалгебре сгенерировать сразу
    несколько слоёв структуры с помощью Free‑монады.
    '''

    def _go(free: Free[A]) -> Fix:
        match free.value:
            case Fix() as fix:                  return fix
            case f if isinstance(f, Functor):   return Fix.i(f.map(_go))
            case a:                             return Fix.i(coalg(a).map(_go))
    
    return Fix.i(coalg(seed).map(_go))


def zygo(alg: ZygoAlgebra, aux_alg: Algebra, fix: Fix) -> A:
    '''
    Зигоморфизм. Параллельно вычисляет основной и вспомогательный результаты.
    Вспомогательный результат доступен основной алгебре через пары (A, B).
    '''

    def _fst(p: Tuple[A, B]) -> A: return p[0]
    def _snd(p: Tuple[A, B]) -> B: return p[1]

    def _aux(_fix: Fix) -> Tuple[A, B]:
        pairs: Functor[Tuple[A, B]] = _fix.o().map(_aux)
        
        b = aux_alg(pairs.map(_snd))
        a = alg(pairs)

        return (a, b)

    return _fst(_aux(fix))

# ---------- стандартные функторы для рекурсивных структур данных ---------- #

@dataclass(frozen = True)
class ListF(Generic[A, T], Functor[T]): 
    ''' Базовый класс для функтора списка: Nil | Cons '''

    pass

@dataclass(frozen = True)
class NilF(ListF[A, T]):
    '''Пустой список '''

    def map(self, f: Callable[[T], U]) -> 'NilF[A, U]':     return NilF()

@dataclass(frozen = True)
class ConsF(ListF[A, T]):
    ''' Ячейка списка: голова (значение) и хвост (рекурсивная позиция) '''

    h: A
    t: T

    def map(self, f: Callable[[T], U]) -> 'ConsF[A, U]':    return ConsF(self.h, f(self.t))


@dataclass(frozen = True)
class TreeF(Generic[A, T], Functor[T]): 
    ''' Базовый класс для функтора бинарного дерева: Empty | Leaf | Node '''

    pass

@dataclass(frozen = True)
class EmptyF(TreeF[A, T]):
    ''' Пустое дерево '''

    def map(self, f: Callable[[T], U]) -> 'EmptyF[A, U]':   return EmptyF()

@dataclass(frozen = True)
class LeafF(TreeF[A, T]):
    ''' Лист дерева, хранящий значение типа A '''

    v: A

    def map(self, f: Callable[[T], U]) -> 'LeafF[A, U]':    return LeafF(self.v)

@dataclass(frozen = True)
class NodeF(TreeF[A, T]):
    ''' Внутренний узел дерева с двумя поддеревьями '''

    l: T
    r: T

    def map(self, f: Callable[[T], U]) -> 'NodeF[A, U]':    return NodeF(f(self.l), f(self.r))

# ---------- конструкторы неподвижных точек рекурсивных структур данных ---------- #

@dataclass(frozen = True)
class FixConstructors:
    ''' Удобные конструкторы для создания Fix-обёрток над стандартными функторами '''
    
    # списки
    @staticmethod
    def nil() -> Fix:
        ''' создаёт пустой список как Fix '''

        return Fix.i(NilF())
    
    @staticmethod
    def cons(head: A, tail: Fix) -> Fix:
        ''' создаёт Fix‑список из головы и хвоста '''

        return Fix.i(ConsF(head, tail))
    
    # деревья
    @staticmethod
    def empty() -> Fix:
        ''' создаёт пустое дерево как Fix '''

        return Fix.i(EmptyF())
    
    @staticmethod
    def leaf(value: A) -> Fix:
        ''' создаёт Fix‑лист с заданным значением '''

        return Fix.i(LeafF(value))
    
    @staticmethod
    def node(left: Fix, right: Fix) -> Fix:
        ''' создаёт Fix‑узел с левым и правым поддеревьями '''

        return Fix.i(NodeF(left, right))