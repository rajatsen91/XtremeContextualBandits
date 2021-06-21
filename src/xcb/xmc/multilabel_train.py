import os
import gc
from tqdm import tqdm
import numpy as np
import scipy.sparse as smat
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.linear_model import Ridge, LogisticRegression
from pathlib import Path
import json
from multiprocessing import cpu_count, Pool

_FUNC = None  # place holder to Pool functions.


def _worker_init(func):
    """Init method to invoke Pool."""
    global _FUNC
    _FUNC = func


def _worker(x):
    """Init function to invoke Pool."""
    return _FUNC(x)


class MultiLabelInstance(object):
    """Class for input data type of model for one level."""

    def __init__(self, X, Y, C, M=None):
        self.X = smat.csr_matrix(X, dtype=np.float32)
        self.Y = smat.csc_matrix(Y, dtype=np.float32)
        self.C = smat.csc_matrix(C, dtype=np.float32)
        if M is None:
            self.M = self.Y.dot(self.C).tocsc()
        else:
            self.M = smat.csc_matrix(M, dtype=np.float32)
        self.X.eliminate_zeros()
        self.Y.eliminate_zeros()
        self.C.eliminate_zeros()
        self.M.eliminate_zeros()
        self.C_csr = self.C.tocsr()

    @property
    def nr_labels(self):
        return self.Y.shape[1]


class MultiLabelSolve(object):
    """Object to hold solution to a multilabel instance."""

    def __init__(self, W, C):
        self.W = smat.csc_matrix(W, dtype=np.float32)
        self.C = smat.csc_matrix(C, dtype=np.float32)

    @property
    def nr_labels(self):
        return self.W.shape[1]

    @property
    def nr_codes(self):
        return self.C.shape[1]

    @property
    def nr_features(self):
        return self.W.shape[0] - 1

    def save(self, model_folder):
        """Save model."""
        smat.save_npz(Path(model_folder, "W.npz"), self.W)
        smat.save_npz(Path(model_folder, "C.npz"), self.C)

    @classmethod
    def load(cls, model_folder):
        """Load model."""
        W = smat.load_npz(Path(model_folder, "W.npz"))
        C = smat.load_npz(Path(model_folder, "C.npz"))
        return cls(W, C)

    @classmethod
    def train(
        cls,
        mli,
        learner="SVC",
        linear_config={"tol": 0.1, "max_iter": 40},
        threshold=0.1,
        threads=cpu_count(),
    ):
        """Train code for mli.

        Parameters:
        ---------
        mli: object
            ml instance
        regression: bool
            if true then regression training is used
        linear_config: dict
            config for liblinear
        threshold: float
            values below this are deleted

        Returns:
            instance of type MultiLabelSolve
        """
        chunks = np.array_split(np.arange(mli.nr_labels), threads)
        results = []
        with Pool(
            processes=threads,
            initializer=_worker_init,
            initargs=(lambda i: cls._train_label(mli, i, learner, linear_config, threshold),),
        ) as pool:
            results = pool.map(_worker, np.arange(mli.nr_labels))
        gc.collect()
        W = smat.hstack(results).tocsc()
        return cls(W, mli.C)

    @classmethod
    def _train_label(self, mli, label, learner, linear_config, threshold):
        positives = set(list(mli.Y[:, label].indices))
        negatives = set(list(mli.M[:, mli.C_csr[label, :].indices[0]].indices)).difference(
            positives
        )
        if len(negatives) == 0:
            negatives = [np.random.choice(mli.X.shape[0])]
        if len(positives) == 0:
            positives = [np.random.choice(mli.X.shape[0])]
        positives = list(positives)
        negatives = list(negatives)
        Xpos = mli.X[positives, :]
        Xneg = mli.X[negatives, :]
        X = smat.vstack([Xpos, Xneg]).tocsr()
        del Xpos, Xneg
        # gc.collect()
        y = [1.0] * len(positives) + [0.0] * len(negatives)
        if learner == "SVR":
            sv = LinearSVR(**linear_config)
        elif learner == "SVC":
            sv = LinearSVC(**linear_config)
        elif learner == "Ridge":
            sv = Ridge(**linear_config)
        elif learner == "Logistic":
            sv = LogisticRegression(**linear_config)
        else:
            raise NotImplementedError("Learner not supported")
        sv = sv.fit(X, y)
        coefs = np.append(sv.coef_.reshape(-1), sv.intercept_)
        coefs[np.abs(coefs) < threshold] = 0.0
        coefs = smat.csc_matrix(coefs.reshape(-1, 1), dtype=np.float32)
        # gc.collect()
        return coefs


class HierarchicalModel(object):
    """Object that trains a model chain for a hierarchical model."""

    def __init__(self, model_chain):
        self.model_chain = model_chain

    def __len__(self):
        return len(self.model_chain)

    @property
    def nr_labels(self):
        return self.model_chain[-1].nr_labels

    @property
    def nr_features(self):
        return self.model_chain[-1].nr_features

    def save(self, folder):
        """Save model."""
        params = {
            "depth": len(self.model_chain),
            "nr_labels": self.nr_labels,
        }
        os.makedirs(folder, exist_ok=True)
        with open(Path(folder, "params.json"), "w") as fp:
            json.dump(params, fp)
        os.makedirs(Path(folder, "ranker"), exist_ok=True)
        for i, m in enumerate(self.model_chain):
            os.makedirs(Path(folder, "ranker", "{}.model".format(i)), exist_ok=True)
            m.save(Path(folder, "ranker", "{}.model".format(i)))

    @classmethod
    def load(cls, folder):
        with open(Path(folder, "params.json"), "r") as fp:
            params = json.load(fp)
        model_chain = []
        for d in range(params["depth"]):
            m = MultiLabelSolve.load(Path(folder, "ranker", "{}.model".format(d)))
            model_chain.append(m)
        return cls(model_chain)

    @classmethod
    def train(
        cls,
        X,
        Y,
        cluster_chain,
        learner="SVC",
        linear_config={"tol": 0.1, "max_iter": 40},
        threshold=0.1,
        threads=cpu_count(),
    ):
        model_chain = []
        for level in tqdm(reversed(range(len(cluster_chain)))):
            C = cluster_chain[level]
            mli = MultiLabelInstance(X, Y, C)
            model = MultiLabelSolve.train(
                mli,
                learner=learner,
                linear_config=linear_config,
                threshold=threshold,
                threads=threads,
            )
            model_chain.insert(0, model)
            Y = mli.M
        return cls(model_chain)
