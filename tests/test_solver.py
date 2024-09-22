from pyomo.contrib.appsi.solvers import Highs
import highspy

def test_highs():
    solver = Highs()
    assert solver.available()

def test_pyhighs():
    h = highspy.Highs()
