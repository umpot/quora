from fastFM import sgd
from fastFM.mcmc import FMClassification, FMRegression

clf = FMClassification(rank=rank, n_iter=n_iter)
clf.fit_predict_proba()