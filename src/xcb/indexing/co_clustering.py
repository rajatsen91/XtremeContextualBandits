import gc
import numpy as np
import scipy.sparse as smat
from sklearn.decomposition import TruncatedSVD
import kmeans1d
from sklearn.preprocessing import normalize
from multiprocessing import cpu_count, Pool

_FUNC = None  # place holder to Pool functions.


def _worker_init(func):
    """Init method to invoke Pool."""
    global _FUNC
    _FUNC = func


def _worker(x):
    """Init function to invoke Pool."""
    return _FUNC(x)


class Node(object):
    def __init__(self, indices, parent, left, right, height):
        self.indices = indices
        self.parent = parent
        self.left = left
        self.right = right
        self.height = height

    def free(self):
        self.indices = None

    def set_left(self, left):
        self.left = left

    def set_right(self, right):
        self.right = right


class HierarchicalCoCluster(object):
    def __init__(self, max_imbalance=0.7, max_leaf_size=100, threads=cpu_count(), verbose=False):
        self.max_imbalance = max_imbalance
        self.max_leaf_size = max_leaf_size
        self.root = None
        self.threads = threads
        self.max_height = 0
        self.verbose = verbose

    def cluster(self, feat_mat):
        feat_mat = smat.csr_matrix(feat_mat)
        feat_mat.data = np.nan_to_num(feat_mat.data)
        self.feat_mat = normalize(feat_mat, norm="l2")
        self.feat_mat.data = np.clip(self.feat_mat.data, a_min=-1e3, a_max=1e3)
        self.root = Node(np.arange(feat_mat.shape[0]), None, None, None, 0)
        leaf_set = [self.root]
        lvl = 0
        while True:
            if self.verbose:
                print("Level: {}".format(lvl))
            ls = [leaf for leaf in leaf_set if len(leaf.indices) > self.max_leaf_size]
            if len(ls) == 0:
                break
            if self.verbose:
                print("# breakable nodes: {}".format(len(ls)))
            results = []
            with Pool(
                processes=self.threads,
                initializer=_worker_init,
                initargs=(lambda i: self._break_node(ls[i]),),
            ) as pool:
                results = pool.map(_worker, np.arange(len(ls)))
            leaf_set = []
            lvl += 1
            for lf, r in zip(ls, results):
                lf.free()
                lf.set_left(r[0])
                lf.set_right(r[1])
                leaf_set += r
                if self.max_height < r[0].height:
                    self.max_height = r[0].height
        if self.verbose:
            print("Getting the chain...")
        return self._get_chain()

    def _get_chain(self):
        current_level = [self.root]
        cmatrices = []
        for level in range(0, self.max_height):
            child = 0
            next_level = []
            data = []
            cr = []
            cc = []
            for i, node in enumerate(current_level):
                if node.left:
                    data.append(1)
                    cc.append(i)
                    cr.append(child)
                    child += 1
                    next_level.append(node.left)
                if node.right:
                    data.append(1)
                    cc.append(i)
                    cr.append(child)
                    child += 1
                    next_level.append(node.right)
                if node.right is None and node.left is None and node.height < self.max_height:
                    data.append(1)
                    cc.append(i)
                    cr.append(child)
                    child += 1
                    node.height += 1
                    next_level.append(node)

            cmatrices.append(
                smat.coo_matrix(
                    (data, (cr, cc)), shape=(len(next_level), len(current_level)), dtype=np.int32
                ).tocsc()
            )
            current_level = next_level
        data = []
        cr = []
        cc = []
        for i, n in enumerate(current_level):
            idx = list(n.indices)
            cr += idx
            data += [1] * len(idx)
            cc += [i] * len(idx)
        cmatrices.append(
            smat.coo_matrix(
                (data, (cr, cc)),
                shape=(self.feat_mat.shape[0], len(current_level)),
                dtype=np.int32,
            ).tocsc()
        )
        return cmatrices

    def _break_node(self, n):
        idx = n.indices
        h = n.height
        n = None
        gc.collect()
        amatrix = self.feat_mat[idx, :]
        amatrix = amatrix[:, amatrix.getnnz(0) > 0]
        id1, id2 = self._cocluster(amatrix)
        left = Node(idx[id1], None, None, None, h + 1)
        right = Node(idx[id2], None, None, None, h + 1)
        return [left, right]

    def _cocluster(self, amatrix):
        d1 = smat.diags(1.0 + np.array(np.sum(amatrix, axis=1)).reshape(-1), format="csr")
        d2 = smat.diags(1.0 + np.array(np.sum(amatrix, axis=0)).reshape(-1), format="csr")
        d1.data[d1.data == 0.0] = 1.0
        d2.data[d2.data == 0.0] = 1.0
        d1.data = 1.0 / np.sqrt(d1.data)
        d2.data = 1.0 / np.sqrt(d2.data)
        amatrix = d1.dot(amatrix).dot(d2).tocsr()
        amatrix.data = np.clip(amatrix.data, a_min=-1e3, a_max=1e3)
        svds = TruncatedSVD(n_components=2, random_state=111, n_iter=2)
        if amatrix.shape[1] > 2:
            umatrix = svds.fit_transform(amatrix)
            decision_vector = np.array(d1.dot(umatrix[:, [1]])).reshape(-1)
        else:
            decision_vector = np.zeros(amatrix.shape[0])
            del amatrix
            gc.collect()
        return self._partition(decision_vector)

    def _partition(self, decision_vector):
        if self.max_imbalance <= 0.5:
            return self._balanced_partition(decision_vector)
        labels, centroids = kmeans1d.cluster(decision_vector, 2)
        labels = np.array(labels)
        c1 = np.where(labels == 0)[0].reshape(-1)
        c2 = np.where(labels == 1)[0].reshape(-1)
        mlen = np.max([c1.shape[0], c2.shape[0]])
        if mlen / decision_vector.shape[0] > self.max_imbalance:
            return self._balanced_partition(decision_vector)
        return c1, c2

    def _balanced_partition(self, decision_vector):
        idx = np.argsort(decision_vector)
        return idx[0 : len(idx) // 2], idx[len(idx) // 2 : :]
