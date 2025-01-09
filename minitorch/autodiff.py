from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
from collections import deque

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    """
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    x1 = list(vals)
    x1[arg] += epsilon

    x0 = list(vals)
    x0[arg] -= epsilon

    return (f(*x1) - f(*x0)) / (epsilon * 2) 


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    scalar_lst = []
    scalar_deq = deque(variable.chain_rule(1.0))
    scalar_set = set([variable.unique_id])
    
    # must use dfs to process topsort
    def dfs(scalar: Scalar):
        if scalar.unique_id in scalar_set:
            return 
        
        if scalar.is_leaf():
            scalar_lst.append(scalar)
            scalar_set.add(scalar.unique_id)
        else:
            for input in scalar.history.inputs:
                if input.is_constant():
                    continue
                dfs(input)
                scalar_lst.append(scalar)
                scalar_set.add(scalar.unique_id)

    return scalar_lst[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    scalar_dict = {variable: deriv}

    scalar_lst = topological_sort(variable)
    for it in scalar_lst:
        if it.is_leaf():
            it.accumulate_derivative(deriv)
        else:
            scalar_lst, value_lst = it.chain_rule()
            for i in range(len(scalar_lst)):
                scalar_dict[scalar_lst[i]] += value_lst[i]


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
