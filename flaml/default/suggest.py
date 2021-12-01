
import numpy as np
from sklearn.neighbors import NearestNeighbors
import logging
from flaml.data import CLASSIFICATION

logger = logging.getLogger(__name__)


def meta_feature(task, X_train, y_train):
    is_classification = task in CLASSIFICATION
    n_row = X_train.shape[0]
    n_feat = X_train.shape[1]
    n_class = len(np.unique(y_train)) if is_classification else 0
    percent_num = X_train.select_dtypes(include=np.number).shape[1] / n_feat
    return (n_row, n_feat, n_class, percent_num)
    

def suggest_config(task, X, y, portfolio):
    assert portfolio["version"] == "default"
    prep = portfolio["preprocessing"]
    feature = meta_feature(task, X ,y)
    feature = (np.array(feature) - np.array(prep["center"])) / np.array(prep["scale"])
    neighbors = portfolio["neighbors"]
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit([x["features"] for x in neighbors])
    dist, ind = nn.kneighbors(feature.reshape(1, -1), return_distance=True)
    logger.info(f"metafeature distance: {dist.item()}")
    ind = int(ind.item())
    choice = int(neighbors[ind]["choice"][0])
    return portfolio["portfolio"][choice]