"""Regression guards against reverting to locally generated CEC instances."""
import numpy as np
import pytest
from acflc.benchmarks import make, BudgetedObjective

def test_f6_uses_published_shift_prefix():
    fn=make('F6',2)
    np.testing.assert_allclose(fn.shift,[81.0232,-48.395])
    np.testing.assert_allclose(fn([[81.0232,-48.395]]),[390.])

def test_exhausted_budget_does_not_allow_free_objective_calls():
    objective=BudgetedObjective(make('F9',2),1)
    objective([[0,0]])
    with pytest.raises(RuntimeError,match='budget exhausted'):
        objective([[1,1]])
