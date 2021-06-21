"""Contextual bandit inference for xtreme  models."""
import json
import gc
import numpy as np
from pathlib import Path
import scipy.sparse as smat

from xcb import core
from multiprocessing import cpu_count, Pool

_FUNC = None  # place holder to Pool functions.


def _worker_init(func):
    """Init method to invoke Pool."""
    global _FUNC
    _FUNC = func


def _worker(x):
    """Init function to invoke Pool."""
    return _FUNC(x)


def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class XLinearCBI(object):
    """[C]ontextual [B]andit [I]nference object for xtreme model."""

    def __init__(self, model_chain, params, num_labels):
        """Initializing class.

        Parameters:
        -----------
        model_chain: obj
            xlinear model chain inference object from core
        params: dict
            parameters for the model
        num_labels: int
            number of labels for the model
        """
        self.model_chain = model_chain
        self.params = params
        self.num_labels = num_labels

    @classmethod
    def load(cls, model_path):
        """Load xfalcon model from path.

        Parameters:
        ----------
        model_path: str
            path to saved xlinear model

        Returns:
        -------
        core.ModelChain() object and parameter dictionary.
        """
        routing_ranker_path = Path(model_path, "routing_model", "ranker")
        regression_ranker_path = Path(model_path, "regression_model", "ranker")
        params = json.load(open(Path(model_path, "routing_model", "params.json"), "r"))
        model_chain = core.ModelChain()
        for d in range(params["depth"]):
            weight = smat.load_npz(Path(routing_ranker_path, "{}.model".format(d), "W.npz"))
            cluster = smat.load_npz(Path(routing_ranker_path, "{}.model".format(d), "C.npz"))
            regression_weight = smat.load_npz(
                Path(regression_ranker_path, "{}.model".format(d), "W.npz")
            )
            model_chain.add_elements(weight, cluster, regression_weight)
            if d == params["depth"] - 1:
                num_labels = cluster.shape[0]
            del weight, cluster
            gc.collect()
        return cls(model_chain, params, num_labels)

    def predict_realtime(
        self,
        x,
        beam_size=10,
        topk=10,
        num_explore=5,
        post_processor="sigmoid",
        combiner="noop",
        explore_in_routing=True,
        explore_strategy="falcon",
        multiplier=-1.0,
        alpha=0.5,
    ):
        """Predict given a sample x.

        Parameters:
        ----------
        x: csr matrix
            input sample in sparse csr format
        beam_size: int
            beam size for search
        topk: int
            get only topk results
        num_explore: int
            number of slots on which we are allowed to explore
        post_processor: str
            process scores using this transform function
        combiner: str
            method to combine scores from previous levels with current scores
        explore_in_routing: str
            explore in non leaf levels if True
        multiplier: float
            factor used to determine the level of falcon exploration.
            higher value means more exploitation.
            negative value means pure greedy inference.
        alpha: float
            gamma in falcon is (multiplier*k)^{alpha}
        """
        if post_processor not in ["noop", "sigmoid", "l3-hinge"]:
            raise NotImplementedError("post_processor not implemented!")
        if combiner not in ["noop", "add", "multiply"]:
            raise NotImplementedError("combiner not implemented!")
        if not isinstance(x, smat.csr_matrix):
            raise ValueError("x should be of type scipy sparse csr_matrix")
        if not (0 < num_explore <= topk):
            raise ValueError("num_explore should be between 0 and topk")
        bias = self.params.get("bias", 1.0)
        if bias is not None:
            x = smat.csr_matrix(smat.hstack([x, [bias]]), dtype=np.float32)
        return self.model_chain.beam_search(
            x,
            beam_size,
            topk,
            num_explore,
            multiplier,
            post_processor,
            combiner,
            explore_in_routing,
            explore_strategy,
            alpha,
        )

    def predict(
        self,
        X,
        beam_size=10,
        topk=10,
        num_explore=5,
        post_processor="sigmoid",
        combiner="noop",
        explore_in_routing=True,
        explore_strategy="falcon",
        multiplier=-1.0,
        alpha=0.5,
        threads=cpu_count(),
        batch_size=10000,
    ):
        """Predict given a sample x.

        Parameters:
        ----------
        x: csr matrix
            input samples in sparse csr format
        beam_size: int
            beam size for search
        topk: int
            get only topk results
        num_explore: int
            number of slots on which we are allowed to explore
        post_processor: str
            process scores using this transform function
        combiner: str
            method to combine scores from previous levels with current scores
        explore_in_routing: str
            explore in non leaf levels if True
        multiplier: float
            factor used to determine the level of falcon exploration.
            higher value means more exploitation.
            negative value means pure greedy inference.
        threads: int
            number of threads ro launch
        batch_size: int
            batch inputs for prediction
        alpha: float
            gamma in falcon is (multiplier*k*t)^{alpha}
        """
        inputs = list(X)
        chunks = _chunks(inputs, batch_size)
        all_preds = []
        all_maps = []
        for c in chunks:
            with Pool(
                processes=threads,
                initializer=_worker_init,
                initargs=(
                    lambda x: self.predict_realtime(
                        x,
                        beam_size=beam_size,
                        topk=topk,
                        num_explore=num_explore,
                        post_processor=post_processor,
                        combiner=combiner,
                        explore_in_routing=explore_in_routing,
                        multiplier=multiplier,
                        explore_strategy=explore_strategy,
                        alpha=alpha,
                    ),
                ),
            ) as pool:
                results = pool.map(_worker, c)
            preds = [r[0] for r in results]
            maps = [r[1] for r in results]
            all_preds += preds
            all_maps += maps
            gc.collect()
        pred_mat = self._convert_to_csr(all_preds)
        return pred_mat, all_maps

    def _convert_to_csr(self, features):
        """
        Helper function to convert dictionary of features to sparse csr_matrix.

        Parameters:
        ----------
        features: list(dictionary) (or sparse csr matrix)
            a sparse matrix represented as list of dictionary, each element of the list is a row.
            Each row's dictionary has indices mapped to values

        Returns:
        -------
        sparse csr matrix.
        """
        if isinstance(features, smat.csr_matrix):
            return features
        dimension = self.num_labels
        data = []
        indices = []
        ptr = 0
        indptr = [ptr]
        for f in features:
            data += [val[1] for val in f]
            indices += [val[0] for val in f]
            ptr += len(f)
            indptr.append(ptr)

        return smat.csr_matrix(
            (data, indices, indptr),
            shape=(len(features), dimension),
            dtype=np.float32,
        )
