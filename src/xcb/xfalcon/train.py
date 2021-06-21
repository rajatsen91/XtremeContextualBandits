"""This is a wrapper for training XFalcon models."""
import gc
import copy
import logging
import os
from pathlib import Path
import scipy.sparse as smat
import numpy as np

from pecos.xmc import LabelEmbeddingFactory
from xcb.indexing import co_clustering
from xcb.xmc import multilabel_train as mt

LOGGER = logging.getLogger(__name__)


class XFalconTrainer(object):
    def __init__(self, routing_model, regression_model):
        """Init class.

        Parameters:
        ----------
        routing_model: xcb  Model
            model for routing
        regression_model: xcb  Model
            model for regression
        """
        self.routing_model = routing_model
        self.regression_model = regression_model

    def save(self, folder):
        """Save method."""
        os.makedirs(folder, exist_ok=True)
        self.routing_model.save(Path(folder, "routing_model"))
        self.regression_model.save(Path(folder, "regression_model"))

    @classmethod
    def load(cls, folder):
        """Load methods."""
        routing_model = mt.HierarchicalModel.load(Path(folder, "routing_model"))
        regression_model = mt.HierarchicalModel.load(Path(folder, "regression_model"))
        return cls(routing_model, regression_model)

    @classmethod
    def train(cls, X, Ys, Ya, Z, train_config, previous_model_path=None):
        """Train Model.

        Parameters:
        ----------
        X: csr matrix
            feature matrix
        Ys: csc matrix
            label matrix encoding arms selected
        Ya: csc matrix
            ground-truth label matrix
        Z : list(list(tuples))
            each row is a mapping of Y_s[i,:] selected arms to chunks.
            The tuple is ((l, c), arm). arm is the selected singlton arm.
            l is the level of the chunk and c is the the node id in the level.
        train_config: dict
            training configurations

        Returns:
        -------
        Trained ranking and regression models
        """
        X = smat.csr_matrix(X, dtype=np.float32)
        if Ys is None:
            Y = Ya
        else:
            Y = Ys.multiply(Ya)
            Y.eliminate_zeros()
        if Z is None:
            Z = [[] for _ in range(X.shape[0])]
        Y = smat.csc_matrix(Y, dtype=np.float32)
        if train_config["mode"] == "full" or not previous_model_path:
            label_feat = LabelEmbeddingFactory.create(Y, X, method="pifa")
            cluster_config = train_config["cluster_config"]
            LOGGER.info("Clustering...")
            hc = co_clustering.HierarchicalCoCluster(**cluster_config)
            cmat = hc.cluster(label_feat)
            del hc
            gc.collect()
        else:
            prev_model = cls.load(previous_model_path)
            cmat = [m.C for m in prev_model.routing_model.model_chain]
        model_config = copy.deepcopy(train_config["model_config"])
        model_config["X"] = X
        model_config["Y"] = Y
        model_config["cluster_chain"] = cmat
        if train_config["mode"] == "full" or not previous_model_path:
            model_config["learner"] = model_config["learner"]["class"]
            model_config["linear_config"] = model_config["linear_config"]["class"]
            routing_model = mt.HierarchicalModel.train(**model_config)
        else:
            routing_model = prev_model.routing_model
        regression_model = cls._train_regression_model(X, Ys, Ya, Z, cmat, Y, train_config)
        return cls(routing_model, regression_model)

    @classmethod
    def _train_last_level(cls, X, Ys, Ya, Y, cmat, train_config):
        """Train last level of regressor"""
        Ys.data = np.ones(shape=Ys.data.shape)
        Y.data = np.ones(shape=Y.data.shape)
        Ys.eliminate_zeros()
        Y.eliminate_zeros()
        Yn = Ys - Y
        Yn.eliminate_zeros()
        label_matrix = smat.csc_matrix(smat.hstack([Y, Yn]), dtype=np.float32)
        r = list(range(2 * Y.shape[1]))
        c = list(range(Y.shape[1])) + list(range(Y.shape[1]))
        v = [1] * (2 * Y.shape[1])
        cluster_matrix = smat.csc_matrix((v, (r, c)), shape=(2 * Y.shape[1], Y.shape[1]))
        problem = mt.MultiLabelInstance(X, label_matrix, cluster_matrix)
        model_config = copy.deepcopy(train_config["model_config"])
        model_config["mli"] = problem
        model_config["learner"] = model_config["learner"]["reg"]
        model_config["linear_config"] = model_config["linear_config"]["reg"]
        mlmodel = mt.MultiLabelSolve.train(**model_config)
        cluster_weights = mlmodel.W[:, 0 : Y.shape[1]]
        if len(Yn.data) == 0:
            cluster_weights.data = np.zeros(shape=cluster_weights.data.shape)
            cluster_weights.eliminate_zeros()
        n_zeros = np.array(np.sum(cluster_weights > 0, axis=0)).reshape(-1)
        cluster_weights = smat.lil_matrix(cluster_weights)
        for j in range(cluster_weights.shape[1]):
            if n_zeros[j] <= 0:
                cluster_weights[-1, j] = -1.0
        cluster_weights = smat.csc_matrix(cluster_weights, dtype=np.float32)
        cluster_weights.eliminate_zeros()
        return mt.MultiLabelSolve(W=cluster_weights, C=cmat[-1])

    @classmethod
    def _train_regression_model(cls, X, Ys, Ya, Z, cmat, Y, train_config):
        """Train regresion model."""
        label_matrix_sizes = []
        cluster_matrices = []
        for C in cmat[:-1]:
            r = list(range(2 * C.shape[0]))
            c = np.arange(2 * C.shape[0]) // 2
            c = c.tolist()
            v = [1] * (2 * C.shape[0])
            cluster_matrices.append(
                smat.csc_matrix((v, (r, c)), shape=(2 * C.shape[0], C.shape[0]))
            )
            label_matrix_sizes.append((X.shape[0], 2 * C.shape[0]))

        rows = [[] for _ in range(len(cmat) - 1)]
        cols = [[] for _ in range(len(cmat) - 1)]
        vals = [[] for _ in range(len(cmat) - 1)]
        for i, Zi in enumerate(Z):
            for ((level, node), leaf_node) in Zi:
                rew = Ya[i, leaf_node]
                rows[level].append(i)
                cols[level].append(2 * node * (rew > 0) + (2 * node + 1) * (rew <= 0))
                vals[level].append(1.0)

        label_matrices = []
        for level in range(len(label_matrix_sizes)):
            label_matrices.append(
                smat.csc_matrix(
                    (vals[level], (rows[level], cols[level])),
                    shape=label_matrix_sizes[level],
                    dtype=np.float32,
                )
            )
        model_chain = []
        model_config = copy.deepcopy(train_config["model_config"])
        model_config["learner"] = model_config["learner"]["reg"]
        model_config["linear_config"] = model_config["linear_config"]["reg"]
        for level in range(len(label_matrix_sizes)):
            problem = mt.MultiLabelInstance(X, label_matrices[level], cluster_matrices[level])
            model_config["mli"] = problem
            mlmodel = mt.MultiLabelSolve.train(**model_config)
            cluster_weights = mlmodel.W[:, ::2]
            n_zeros = np.array(np.sum(cluster_weights > 0, axis=0)).reshape(-1)
            for j in range(cluster_weights.shape[1]):
                if n_zeros[j] <= 0:
                    cluster_weights[-1, j] = -1.0
            cluster_weights.eliminate_zeros()
            new_model = mt.MultiLabelSolve(W=cluster_weights, C=cmat[level])
            model_chain.append(new_model)
        ranker = cls._train_last_level(X, Ys, Ya, Y, cmat, train_config)
        model_chain.append(ranker)
        regression_model = mt.HierarchicalModel(model_chain)
        return regression_model
