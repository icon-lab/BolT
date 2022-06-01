import numpy as np
import torch
from sklearn.covariance import LedoitWolf

# corrcoef based on
# https://github.com/pytorch/pytorch/issues/1254
#and
# https://github.com/egyptdj/stagin/blob/main/util/bold.py
def corrcoef(x):

    """
        x.shape = (#Rois, T)
    """

    x = torch.tensor(x)

    mean_x = torch.mean(x, 1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, -1.0, 1.0)

    c = np.nan_to_num(c.numpy(), 0)

    return c

# in our experience this works worse for the SVM classification performance
def ledoit_wolf_corrcoef(x):
    """
        x.shape = (#Rois, T)
    """

    cov = LedoitWolf().fit(x.T)
    return cov.covariance_

