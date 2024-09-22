from pyomo.environ import SolverFactory

def test_highs():
    assert SolverFactory('highs').available()
