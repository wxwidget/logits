"""Run from an installed LogitLab package: python examples/train.py."""
import numpy as np
from scipy import sparse
from logitlab import LogisticRegression

X = np.array([[-2., 0.], [-1., 1.], [1., 0.], [2., 1.]])
y = np.array([0., 0., 1., 1.])
for features in (X, sparse.csr_matrix(X)):
    model = LogisticRegression(solver="lbfgs", l2=.1).fit(features, y)
    print(model.status_, model.predict(features), model.history_[-1]["loss"])
