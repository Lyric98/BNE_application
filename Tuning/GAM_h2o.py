import h2o
from h2o.estimators.gam import H2OGeneralizedAdditiveEstimator
h2o.init()

# create frame knots
knots1 = [-1.99905699, -0.98143075, 0.02599159, 1.00770987, 1.99942290]
frameKnots1 = h2o.H2OFrame(python_obj=knots1)
knots2 = [-1.999821861, -1.005257990, -0.006716042, 1.002197392, 1.999073589]
frameKnots2 = h2o.H2OFrame(python_obj=knots2)
knots3 = [-1.999675688, -0.979893796, 0.007573327,1.011437347, 1.999611676]
frameKnots3 = h2o.H2OFrame(python_obj=knots3)