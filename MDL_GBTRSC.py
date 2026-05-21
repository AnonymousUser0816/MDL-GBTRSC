"""
    Ttile: Minimum Description Length based Granular-Ball Tree Regularization for Spectral Clustering
    Author: Zeqiang Xian
    Date: 2026/4/28 17:27:33
    Description: The algorithm first grows an MDL granular-ball tree under a graph-continuity regularized split code. The stable leaf balls then regularize a sample affinity
                graph through their local coding scales. Version 1.3 uses reciprocal continuity evidence for tree growth and an MDL-consistent reciprocal/completed graph selection for final partitioning.
    Version: V3.0
    Dependencies: numpy, scipy, scikit-learn.
"""

from __future__ import annotations
import heapq
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.io import arff, loadmat
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

EPS = 1e-12
LOG_TWO = math.log(2.0)
BINARY_MODEL_CODE = LOG_TWO
LOG_2PI = math.log(2.0 * math.pi)

def _validate_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[0] < 2:
        raise ValueError("X must contain at least two samples")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values")
    return X


def _relabel(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    uniq = np.unique(labels)
    mapping = {int(v): i for i, v in enumerate(uniq)}
    return np.asarray([mapping[int(v)] for v in labels], dtype=int)


def _entropy_from_counts(counts: Sequence[int], total: Optional[int] = None) -> float:
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    total = int(np.sum(counts)) if total is None else int(total)
    p = counts / float(max(total, 1))
    return float(-np.sum(p * np.log(np.maximum(p, EPS))))


def _sse_from_stats(n: int, s: np.ndarray, ss: float) -> float:
    if n <= 0:
        return math.inf
    return max(float(ss - float(np.dot(s, s)) / float(n)), 0.0)


def _prefix_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sums = np.cumsum(X, axis=0)
    sumsqs = np.cumsum(np.einsum("ij,ij->i", X, X))
    return sums, sumsqs


def _sse_vec(ns: np.ndarray, sums: np.ndarray, sumsqs: np.ndarray) -> np.ndarray:
    ns = np.maximum(np.asarray(ns, dtype=float), 1.0)
    return np.maximum(sumsqs - np.einsum("ij,ij->i", sums, sums) / ns, 0.0)


def _majority(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=int)
    return int(np.argmax(np.bincount(values)))


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _relabel(np.asarray(y_true, dtype=int))
    y_pred = _relabel(np.asarray(y_pred, dtype=int))
    w = np.zeros((int(y_pred.max()) + 1, int(y_true.max()) + 1), dtype=int)
    np.add.at(w, (y_pred, y_true), 1)
    row, col = linear_sum_assignment(-w)
    return float(w[row, col].sum() / max(len(y_true), 1))


@dataclass
class BallNode:
    indices: np.ndarray
    level: int
    node_id: int
    parent_id: Optional[int] = None
    metadata: Dict = field(default_factory=dict)

    n: int = 0
    d: int = 0
    center: Optional[np.ndarray] = None
    radius: float = 0.0
    sse: float = 0.0
    sigma2: float = 0.0
    leaf_cost: float = math.inf
    leaf_model: str = "iso"
    direction: Optional[np.ndarray] = None
    children: List["BallNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class MDLGranularBallTreeRegularizedSpectralClustering:
    """
    n_clusters:
        Target number of clusters. If None, it is estimated by eigengap on the learned affinity graph.
    scale_data:
        Whether to min-max normalize the input before fitting.
    use_subspace_ball:
        Whether to compare isotropic and PCA-subspace leaf codes.
    compute_partition_code:
        Whether to compute the final reporting objective after fitting.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        *,
        scale_data: bool = True,
        use_subspace_ball: bool = True,
        compute_partition_code: bool = False,
    ) -> None:
        self.n_clusters = None if n_clusters is None else int(n_clusters)
        self.scale_data = bool(scale_data)
        self.use_subspace_ball = bool(use_subspace_ball)
        self.compute_partition_code = bool(compute_partition_code)

        self.scaler_: Optional[MinMaxScaler] = None
        self.X_: Optional[np.ndarray] = None
        self.n_samples_: int = 0
        self.n_features_: int = 0
        self.sigma2_floor_: float = EPS
        self.use_subspace_ball_: bool = bool(use_subspace_ball)

        self.root_: Optional[BallNode] = None
        self.nodes_: Dict[int, BallNode] = {}
        self.leaf_nodes_: List[BallNode] = []
        self.sample_to_ball_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.ball_labels_: Optional[np.ndarray] = None
        self.base_graph_: Optional[csr_matrix] = None
        self.graph_: Optional[csr_matrix] = None
        self.partition_code_: float = math.nan
        self.history_: List[Dict] = []
        self.diagnostics_: Dict = {}

        self._knn_order_: Optional[int] = None
        self._knn_dist_: Optional[np.ndarray] = None
        self._knn_ind_: Optional[np.ndarray] = None
        self._knn_local_scale_: Optional[np.ndarray] = None
        self._split_mark_: Optional[np.ndarray] = None
        self._split_stamp_: int = 0

        self._graph_rows_: Optional[np.ndarray] = None
        self._graph_cols_: Optional[np.ndarray] = None
        self._graph_bridge_: Optional[np.ndarray] = None
        self._graph_base_vals_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MDLGranularBallTreeRegularizedSpectralClustering":
        del y
        t0 = time.time()
        X = _validate_X(X)
        n, d = X.shape
        if self.n_clusters is not None and not (1 <= self.n_clusters <= n):
            raise ValueError("n_clusters must be None or an integer in [1, n_samples]")

        self.n_samples_, self.n_features_ = int(n), int(d)
        self.use_subspace_ball_ = bool(self.use_subspace_ball)
        self.scaler_ = MinMaxScaler() if self.scale_data else None
        self.X_ = self.scaler_.fit_transform(X) if self.scaler_ is not None else X.copy()

        self.nodes_.clear()
        self.leaf_nodes_ = []
        self.history_ = []
        self.diagnostics_ = {}
        self._split_mark_ = np.zeros(n, dtype=np.int32)
        self._split_stamp_ = 0
        self._knn_order_ = None
        self._knn_dist_ = None
        self._knn_ind_ = None
        self._knn_local_scale_ = None
        self._graph_rows_ = None
        self._graph_cols_ = None
        self._graph_bridge_ = None
        self._graph_base_vals_ = None

        graph_order = self._graph_order()
        self._prepare_neighbor_cache(graph_order)
        self.sigma2_floor_ = self._estimate_resolution_variance()
        self.base_graph_ = self._build_base_continuity_graph(graph_order)

        self._build_tree()
        self._build_sample_to_ball()
        self.graph_ = self._build_tree_regularized_graph(graph_order)

        k = self._estimate_k_by_eigengap(self.graph_) if self.n_clusters is None else int(self.n_clusters)
        self.labels_ = self._partition_graph(self.graph_, k)
        self.ball_labels_ = self._labels_to_ball_labels(self.labels_)
        self.partition_code_ = self._partition_total_code(self.labels_) if self.compute_partition_code else math.nan

        self.diagnostics_.update({
            "runtime_sec": float(time.time() - t0),
            "n_samples": int(n),
            "n_features": int(d),
            "n_tree_nodes": int(len(self.nodes_)),
            "n_leaf_balls": int(len(self.leaf_nodes_)),
            "n_clusters": int(len(np.unique(self.labels_))),
            "partition_code": float(self.partition_code_),
            "use_subspace_ball": bool(self.use_subspace_ball_),
        })
        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y=y)
        assert self.labels_ is not None
        return self.labels_.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_ is None or self.labels_ is None:
            raise RuntimeError("The model has not been fitted")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xw = self.scaler_.transform(X) if self.scaler_ is not None else X.copy()

        centers = np.vstack([b.center for b in self.leaf_nodes_])
        sigmas = np.array([max(b.sigma2, self.sigma2_floor_) for b in self.leaf_nodes_])
        sq = np.sum((Xw[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        nearest_ball = np.argmin(sq / np.maximum(sigmas[None, :], EPS), axis=1)

        ball_labels = np.zeros(len(self.leaf_nodes_), dtype=int)
        for bi, b in enumerate(self.leaf_nodes_):
            ball_labels[bi] = _majority(self.labels_[b.indices])
        return ball_labels[nearest_ball]

    def get_ball_summary(self) -> List[Dict]:
        return [
            {
                "ball": int(i),
                "n": int(b.n),
                "level": int(b.level),
                "radius": float(b.radius),
                "sigma2": float(b.sigma2),
                "leaf_model": str(b.leaf_model),
                "leaf_cost": float(b.leaf_cost),
            }
            for i, b in enumerate(self.leaf_nodes_)
        ]

    def materialize_partition_code(self) -> float:
        if self.labels_ is None or self.graph_ is None:
            raise RuntimeError("The model has not been fitted")
        if not np.isfinite(self.partition_code_):
            self.partition_code_ = self._partition_total_code(self.labels_)
            self.diagnostics_["partition_code"] = float(self.partition_code_)
        return float(self.partition_code_)

    # ------------------------- tree construction -------------------------
    def _build_tree(self) -> None:
        assert self.X_ is not None
        root = BallNode(np.arange(self.n_samples_, dtype=int), level=0, node_id=0)
        self.root_ = root
        self.nodes_[0] = root
        self._refresh(root)

        next_id = 1
        leaf_count = 1
        queue: List[Tuple[float, int, int, Dict]] = []
        counter = 0

        root_res = self._evaluate_split(root)
        self.history_.append(root_res)
        if root_res["winner"] == "M2" and root_res["split"] is not None:
            heapq.heappush(queue, (-root_res["gain"], counter, root.node_id, root_res))
            counter += 1

        while queue:
            neg_gain, _, node_id, res = heapq.heappop(queue)
            if -float(neg_gain) <= EPS:
                break

            node = self.nodes_.get(node_id)
            if node is None or not node.is_leaf or res["split"] is None:
                continue

            left_idx, right_idx = res["split"]
            if len(left_idx) == 0 or len(right_idx) == 0:
                continue

            left = BallNode(left_idx, node.level + 1, next_id, parent_id=node.node_id)
            next_id += 1
            right = BallNode(right_idx, node.level + 1, next_id, parent_id=node.node_id)
            next_id += 1

            self._refresh(left, res.get("left_leaf_info"))
            self._refresh(right, res.get("right_leaf_info"))
            node.children = [left, right]
            self.nodes_[left.node_id] = left
            self.nodes_[right.node_id] = right
            leaf_count += 1

            for child in (left, right):
                child_res = self._evaluate_split(child)
                self.history_.append(child_res)
                if child_res["winner"] == "M2" and child_res["split"] is not None:
                    heapq.heappush(queue, (-child_res["gain"], counter, child.node_id, child_res))
                    counter += 1

        self.leaf_nodes_ = self._collect_leaves(root)

    def _refresh(self, node: BallNode, leaf_info: Optional[Tuple[float, str, Optional[np.ndarray]]] = None) -> None:
        assert self.X_ is not None
        X = self.X_[node.indices]
        node.n, node.d = X.shape
        s = np.sum(X, axis=0)
        ss = float(np.einsum("ij,ij->", X, X))
        node.center = s / float(node.n)
        node.sse = _sse_from_stats(node.n, s, ss)
        node.sigma2 = max(node.sse / max(float(node.n * node.d), 1.0), self.sigma2_floor_)
        diff = X - node.center
        node.radius = float(math.sqrt(max(float(np.max(np.einsum("ij,ij->i", diff, diff))), EPS)))
        if leaf_info is None:
            node.leaf_cost, node.leaf_model, node.direction = self._leaf_code(X)
        else:
            node.leaf_cost = float(leaf_info[0])
            node.leaf_model = str(leaf_info[1])
            node.direction = None if leaf_info[2] is None else np.asarray(leaf_info[2], dtype=float).copy()

    def _collect_leaves(self, node: BallNode) -> List[BallNode]:
        if node.is_leaf:
            return [node]
        leaves: List[BallNode] = []
        for child in node.children:
            leaves.extend(self._collect_leaves(child))
        return leaves

    def _n_min(self, n: int, d: int) -> int:
        if n <= 3:
            return 1
        denom = math.log(max(math.sqrt(float(d + 2)), 1.0 + EPS))
        val = min(math.sqrt(float(n)) / max(denom, EPS), float(d + 2))
        return int(min(max(2, math.ceil(val)), max(1, n // 2)))

    def _evaluate_split(self, node: BallNode) -> Dict:
        assert self.X_ is not None
        n, d = self.X_[node.indices].shape
        n_min = self._n_min(n, d)
        retain = node.leaf_cost + BINARY_MODEL_CODE
        best, best_split, best_order, child_leaf_info = math.inf, None, None, None
        if n >= 2 * n_min:
            best, best_split, best_order, child_leaf_info = self._best_threshold_split(node, n_min)
        gain = float(retain - best)
        winner = "M2" if gain > EPS else "M1"
        left_leaf_info = child_leaf_info[0] if winner == "M2" and child_leaf_info is not None else None
        right_leaf_info = child_leaf_info[1] if winner == "M2" and child_leaf_info is not None else None
        return {
            "node_id": int(node.node_id),
            "n": int(n),
            "d": int(d),
            "n_min": int(n_min),
            "L_M1": float(retain),
            "L_M2": float(best),
            "gain": float(max(gain, 0.0)),
            "winner": winner,
            "ordering": best_order,
            "split": best_split if winner == "M2" else None,
            "left_leaf_info": left_leaf_info,
            "right_leaf_info": right_leaf_info,
        }

    def _best_threshold_split(
        self, node: BallNode, n_min: int
    ) -> Tuple[float, Optional[Tuple[np.ndarray, np.ndarray]], Optional[str], Optional[Tuple[Tuple[float, str, Optional[np.ndarray]], Tuple[float, str, Optional[np.ndarray]]]]]:
        assert self.X_ is not None
        X = self.X_[node.indices]
        n, d = X.shape
        valid = np.arange(n_min, n - n_min + 1, dtype=int)
        if valid.size == 0:
            return math.inf, None, None, None

        orderings = self._candidate_orderings(X, node)
        if not orderings:
            return math.inf, None, None, None

        model_code = BINARY_MODEL_CODE
        ordering_code = math.log(len(orderings) + 1.0)
        cut_code = math.log(len(valid) + 1.0)
        n_refine = int(max(2, math.ceil(math.sqrt(n))))

        best = math.inf
        best_split = None
        best_name = None
        best_leaf_info = None

        for name, score in orderings:
            order = np.argsort(score, kind="mergesort")
            Xs = X[order]
            Is = node.indices[order]
            psum, psumsq = _prefix_stats(Xs)
            total_sum = psum[-1]
            total_ss = float(psumsq[-1])

            left_sum = psum[valid - 1]
            left_ss = psumsq[valid - 1]
            right_sum = total_sum - left_sum
            right_ss = total_ss - left_ss
            n1 = valid.astype(float)
            n2 = (n - valid).astype(float)

            approx = (
                self._iso_code_array(n1, d, _sse_vec(n1, left_sum, left_ss))
                + self._iso_code_array(n2, d, _sse_vec(n2, right_sum, right_ss))
                + model_code
                + ordering_code
                + cut_code
            )
            cut_by_valid = self._ordered_split_cut_codes(Is, valid)
            top = np.argsort(approx, kind="mergesort")[: min(valid.size, n_refine)]

            for pos in top:
                k = int(valid[int(pos)])
                left_idx = Is[:k].copy()
                right_idx = Is[k:].copy()
                left_leaf = self._leaf_code(Xs[:k])
                right_leaf = self._leaf_code(Xs[k:])
                cost = (
                    left_leaf[0]
                    + right_leaf[0]
                    + float(cut_by_valid[int(pos)])
                    + model_code
                    + ordering_code
                    + cut_code
                )
                if cost < best:
                    best = float(cost)
                    best_split = (left_idx, right_idx)
                    best_name = name
                    best_leaf_info = (left_leaf, right_leaf)

        return best, best_split, best_name, best_leaf_info

    def _candidate_orderings(self, X: np.ndarray, node: BallNode) -> List[Tuple[str, np.ndarray]]:
        orderings: List[Tuple[str, np.ndarray]] = []
        seen: List[np.ndarray] = []

        def add(name: str, score: np.ndarray) -> None:
            score = np.asarray(score, dtype=float).ravel()
            if score.size != len(X) or float(np.ptp(score)) <= EPS:
                return
            z = (score - np.mean(score)) / max(float(np.std(score)), EPS)
            for old in seen:
                if abs(float(np.dot(old, z)) / max(len(z), 1)) > 1.0 - 1e-8:
                    return
            seen.append(z)
            orderings.append((name, score))

        for i, direction in enumerate(self._candidate_directions(X, node)):
            add(f"linear_{i}", X[:, 0] if X.shape[1] == 1 else X @ direction)
        if node.center is not None:
            diff = X - node.center
            add("radial", np.einsum("ij,ij->i", diff, diff))
        return orderings

    def _candidate_directions(self, X: np.ndarray, node: BallNode) -> List[np.ndarray]:
        directions: List[np.ndarray] = []
        n, d = X.shape

        def add(v: np.ndarray) -> None:
            v = np.asarray(v, dtype=float).ravel()
            if v.size != d:
                return
            norm = float(np.linalg.norm(v))
            if norm <= EPS:
                return
            v = v / norm
            j = int(np.argmax(np.abs(v)))
            if v[j] < 0:
                v = -v
            for old in directions:
                if abs(float(np.dot(old, v))) > 1.0 - 1e-8:
                    return
            directions.append(v)

        add(node.direction if node.direction is not None else self._first_pc(X))
        if n >= 2:
            i = int(np.argmax(np.sum((X - X[0]) ** 2, axis=1)))
            j = int(np.argmax(np.sum((X - X[i]) ** 2, axis=1)))
            add(X[j] - X[i])
        e = np.zeros(d)
        e[int(np.argmax(np.var(X, axis=0)))] = 1.0
        add(e)
        return directions

    # ------------------------------ MDL codes ------------------------------
    def _leaf_code(self, X: np.ndarray) -> Tuple[float, str, Optional[np.ndarray]]:
        iso = self._iso_code(X)
        if X.shape[1] <= 1:
            return iso, "iso", None
        if not self.use_subspace_ball_:
            return iso, "iso", self._first_pc(X)
        sub, direction = self._subspace_code(X)
        return (float(sub + BINARY_MODEL_CODE), "subspace", direction) if sub < iso else (float(iso + BINARY_MODEL_CODE), "iso", direction)

    def _iso_code(self, X: np.ndarray) -> float:
        n, d = X.shape
        s = np.sum(X, axis=0)
        ss = float(np.einsum("ij,ij->", X, X))
        sse = _sse_from_stats(n, s, ss)
        sigma2 = max(sse / max(float(n * d), 1.0), self.sigma2_floor_)
        return float(0.5 * n * d * (LOG_2PI + 1.0 + math.log(sigma2)) + 0.5 * (d + 1) * math.log(max(n, 2)))

    def _iso_code_array(self, ns: np.ndarray, d: int, sse: np.ndarray) -> np.ndarray:
        ns = np.maximum(np.asarray(ns, dtype=float), 1.0)
        sigma2 = np.maximum(sse / np.maximum(ns * float(d), 1.0), self.sigma2_floor_)
        return 0.5 * ns * float(d) * (LOG_2PI + 1.0 + np.log(sigma2)) + 0.5 * (d + 1) * np.log(np.maximum(ns, 2.0))

    def _subspace_code(self, X: np.ndarray) -> Tuple[float, np.ndarray]:
        n, d = X.shape
        if n <= 1:
            return self._iso_code(X), self._first_pc(X)

        Z = X - np.mean(X, axis=0, keepdims=True)
        try:
            if d <= n:
                mat = Z.T @ Z
                eig_all, vecs = np.linalg.eigh(mat)
                order = np.argsort(eig_all)[::-1]
                eig = eig_all[order]
                direction = vecs[:, int(order[0])].copy()
            else:
                mat = Z @ Z.T
                eig_all, uvecs = np.linalg.eigh(mat)
                order = np.argsort(eig_all)[::-1]
                eig = eig_all[order]
                direction = Z.T @ uvecs[:, int(order[0])]
        except Exception:
            var = np.var(Z, axis=0)
            order = np.argsort(var)[::-1]
            eig = var[order] * float(n)
            direction = np.zeros(d)
            direction[int(order[0])] = 1.0

        if eig.size < d:
            eig = np.r_[eig, np.zeros(d - eig.size)]
        eig = np.maximum(eig[:d], 0.0)
        csum = np.r_[0.0, np.cumsum(eig)]
        total = float(csum[-1])

        best = math.inf
        for q in range(d + 1):
            code = 0.0
            if q > 0:
                var_par = max(float(csum[q]) / max(float(n * q), 1.0), self.sigma2_floor_)
                code += 0.5 * n * q * (LOG_2PI + 1.0 + math.log(var_par))
            if d - q > 0:
                var_perp = max(float(total - csum[q]) / max(float(n * (d - q)), 1.0), self.sigma2_floor_)
                code += 0.5 * n * (d - q) * (LOG_2PI + 1.0 + math.log(var_perp))
            k_center = d
            k_var = (1 if q > 0 else 0) + (1 if d - q > 0 else 0)
            k_basis = q * max(d - q, 0)
            best = min(best, float(code + 0.5 * (k_center + k_var + k_basis) * math.log(max(n, 2)) + math.log(d + 1.0)))

        norm = float(np.linalg.norm(direction))
        if norm > EPS:
            direction = direction / norm
        return best, direction

    def _first_pc(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        if d == 1:
            return np.ones(1)
        Z = X - np.mean(X, axis=0, keepdims=True)
        try:
            if d <= n:
                vals, vecs = np.linalg.eigh(Z.T @ Z)
                v = vecs[:, int(np.argmax(vals))]
            else:
                vals, vecs = np.linalg.eigh(Z @ Z.T)
                v = Z.T @ vecs[:, int(np.argmax(vals))]
        except Exception:
            v = np.zeros(d)
            v[int(np.argmax(np.var(Z, axis=0)))] = 1.0
        norm = float(np.linalg.norm(v))
        v = v / norm if norm > EPS else np.r_[1.0, np.zeros(d - 1)]
        j = int(np.argmax(np.abs(v)))
        return -v if v[j] < 0 else v

    # ------------------------------ graph code ------------------------------
    def _graph_order(self) -> int:
        if self.n_samples_ <= 2:
            return 1
        k = math.ceil(math.log2(self.n_samples_ + 1) + math.sqrt(max(self.n_features_, 1)))
        return int(min(max(2, k), self.n_samples_ - 1))

    def _refresh_graph_edge_cache(self) -> None:
        assert self.X_ is not None
        assert self._knn_dist_ is not None and self._knn_ind_ is not None and self._knn_local_scale_ is not None
        n = self.X_.shape[0]
        k = int(self._knn_order_ or 0)
        if n <= 1 or k <= 0:
            self._graph_rows_ = np.empty(0, dtype=int)
            self._graph_cols_ = np.empty(0, dtype=int)
            self._graph_bridge_ = np.empty((n, 0), dtype=float)
            self._graph_base_vals_ = np.empty(0, dtype=float)
            return

        rows = np.repeat(np.arange(n, dtype=int), k)
        cols = self._knn_ind_.ravel()
        denom = np.maximum(self._knn_local_scale_[:, None] * self._knn_local_scale_[self._knn_ind_], EPS)
        bridge = 0.5 * (self._knn_dist_ * self._knn_dist_) / denom
        vals = np.exp(-np.clip(bridge, 0.0, 60.0)).ravel()
        vals = np.minimum(np.maximum(vals, EPS), 1.0 - EPS)

        self._graph_rows_ = rows
        self._graph_cols_ = cols
        self._graph_bridge_ = bridge
        self._graph_base_vals_ = vals

    def _prepare_neighbor_cache(self, graph_order: Optional[int] = None) -> None:
        assert self.X_ is not None
        n = self.X_.shape[0]
        if n <= 1:
            self._knn_order_ = 0
            self._knn_dist_ = np.empty((n, 0), dtype=float)
            self._knn_ind_ = np.empty((n, 0), dtype=int)
            self._knn_local_scale_ = np.ones(n, dtype=float)
            self._refresh_graph_edge_cache()
            return

        k = int(graph_order) if graph_order is not None else self._graph_order()
        k = int(min(max(1, k), n - 1))
        if self._knn_order_ == k and self._knn_dist_ is not None and self._knn_ind_ is not None:
            if self._graph_rows_ is None or self._graph_cols_ is None or self._graph_bridge_ is None or self._graph_base_vals_ is None:
                self._refresh_graph_edge_cache()
            return

        nn = NearestNeighbors(n_neighbors=k + 1).fit(self.X_)
        dist, ind = nn.kneighbors(self.X_, return_distance=True)
        self._knn_order_ = k
        self._knn_dist_ = np.asarray(dist[:, 1:], dtype=float)
        self._knn_ind_ = np.asarray(ind[:, 1:], dtype=int)
        self._knn_local_scale_ = np.maximum(self._knn_dist_[:, -1], EPS)
        self._refresh_graph_edge_cache()

    def _estimate_resolution_variance(self) -> float:
        assert self.X_ is not None
        n, d = self.X_.shape
        if n <= 2:
            return float(max(np.mean(np.var(self.X_, axis=0)) / max(n, 1), EPS))
        if self._knn_dist_ is not None and self._knn_dist_.shape[1] >= 1:
            sq = self._knn_dist_[:, 0] ** 2
            sq = sq[np.isfinite(sq) & (sq > EPS)]
            if sq.size > 0:
                return float(max(np.median(sq) / max(d, 1), EPS))
        return float(max(np.mean(np.var(self.X_, axis=0)) / max(n, 1), EPS))

    def _build_base_continuity_graph(self, graph_order: Optional[int] = None) -> csr_matrix:
        """
            Build the reciprocal continuity graph used by the local MDL split code.
        """
        assert self.X_ is not None
        n = self.X_.shape[0]
        if n <= 1:
            return csr_matrix((n, n))

        self._prepare_neighbor_cache(graph_order)
        assert self._graph_rows_ is not None and self._graph_cols_ is not None and self._graph_base_vals_ is not None
        k = int(self._knn_order_)
        directed = coo_matrix((self._graph_base_vals_, (self._graph_rows_, self._graph_cols_)), shape=(n, n)).tocsr()
        graph = directed.minimum(directed.T)
        graph.setdiag(0.0)
        graph.eliminate_zeros()
        n_comp, _ = connected_components(graph, directed=False)
        self.diagnostics_["base_graph"] = self._graph_diagnostics(graph, k, n_comp)
        self.diagnostics_["base_graph"]["symmetrization"] = "reciprocal"
        return graph

    def _ordered_split_cut_codes(self, ordered_indices: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """
            Return graph-cut code for every prefix split in "valid".
        """
        if self.base_graph_ is None:
            return np.zeros(valid.size, dtype=float)
        ordered_indices = np.asarray(ordered_indices, dtype=int)
        valid = np.asarray(valid, dtype=int)
        n_local = int(ordered_indices.size)
        if n_local <= 1 or valid.size == 0:
            return np.zeros(valid.size, dtype=float)

        pos = np.full(self.n_samples_, -1, dtype=np.int32)
        pos[ordered_indices] = np.arange(n_local, dtype=np.int32)
        diff = np.zeros(n_local + 1, dtype=float)
        graph = self.base_graph_

        for i in ordered_indices:
            i = int(i)
            p = int(pos[i])
            start, end = graph.indptr[i], graph.indptr[i + 1]
            nbr = graph.indices[start:end]
            if nbr.size == 0:
                continue
            q = pos[nbr]
            mask = q > p  # each symmetric reciprocal edge is counted once
            if not np.any(mask):
                continue
            q = q[mask].astype(np.int32, copy=False)
            w = graph.data[start:end][mask]
            costs = -np.log(np.maximum(1.0 - w, EPS))
            diff[p + 1] += float(np.sum(costs))
            np.add.at(diff, q + 1, -costs)

        return np.cumsum(diff)[valid]

    def _split_graph_cut_code(self, left_idx: np.ndarray, right_idx: np.ndarray) -> float:
        if self.base_graph_ is None:
            return 0.0
        left_idx = np.asarray(left_idx, dtype=int)
        right_idx = np.asarray(right_idx, dtype=int)
        if left_idx.size == 0 or right_idx.size == 0:
            return math.inf

        if self._split_mark_ is None or self._split_mark_.size != self.n_samples_:
            self._split_mark_ = np.zeros(self.n_samples_, dtype=np.int32)
            self._split_stamp_ = 0
        self._split_stamp_ += 1
        if self._split_stamp_ >= np.iinfo(np.int32).max:
            self._split_mark_.fill(0)
            self._split_stamp_ = 1

        scan_idx, mark_idx = (left_idx, right_idx) if left_idx.size <= right_idx.size else (right_idx, left_idx)
        self._split_mark_[mark_idx] = self._split_stamp_

        graph = self.base_graph_
        cut = 0.0
        for i in scan_idx:
            start, end = graph.indptr[int(i)], graph.indptr[int(i) + 1]
            nbr = graph.indices[start:end]
            if nbr.size == 0:
                continue
            mask = self._split_mark_[nbr] == self._split_stamp_
            if np.any(mask):
                cut += float(np.sum(-np.log(np.maximum(1.0 - graph.data[start:end][mask], EPS))))
        return float(cut)

    def _build_sample_to_ball(self) -> None:
        sample_to_ball = np.full(self.n_samples_, -1, dtype=int)
        for bi, ball in enumerate(self.leaf_nodes_):
            sample_to_ball[ball.indices] = bi
        if np.any(sample_to_ball < 0):
            raise RuntimeError("Some samples are not covered by leaf balls")
        self.sample_to_ball_ = sample_to_ball

    def _build_tree_regularized_graph(self, graph_order: Optional[int] = None) -> csr_matrix:
        """
            Build the final tree-regularized graph.
        """
        assert self.sample_to_ball_ is not None
        n = self.n_samples_
        if n <= 1:
            return csr_matrix((n, n))

        self._prepare_neighbor_cache(graph_order)
        assert self._knn_ind_ is not None and self._graph_rows_ is not None and self._graph_cols_ is not None and self._graph_bridge_ is not None
        k = int(self._knn_order_)
        ind = self._knn_ind_

        ball_sigma = np.array([max(b.sigma2, self.sigma2_floor_) for b in self.leaf_nodes_])
        ball_radius = np.array([max(b.radius, EPS) for b in self.leaf_nodes_])
        ball_scale = ball_radius + np.sqrt(np.maximum(ball_sigma, EPS)) + EPS

        bi = self.sample_to_ball_[:, None]
        bj = self.sample_to_ball_[ind]
        scale_code = np.abs(np.log(np.maximum(ball_scale[bi], EPS) / np.maximum(ball_scale[bj], EPS)))
        base_code = self._graph_bridge_ + scale_code
        vals = np.exp(-np.clip(base_code, 0.0, 60.0)).ravel()
        vals = np.minimum(np.maximum(vals, EPS), 1.0 - EPS)

        directed = coo_matrix((vals, (self._graph_rows_, self._graph_cols_)), shape=(n, n)).tocsr()

        reciprocal = directed.minimum(directed.T)
        reciprocal.setdiag(0.0)
        reciprocal.eliminate_zeros()
        n_comp_reciprocal, _ = connected_components(reciprocal, directed=False)

        completed = directed.maximum(directed.T)
        completed.setdiag(0.0)
        completed.eliminate_zeros()
        n_comp_completed, _ = connected_components(completed, directed=False)

        bridge_active = False
        if (
            self.n_clusters is not None
            and int(n_comp_reciprocal) == int(n_comp_completed)
            and int(n_comp_completed) < int(self.n_clusters)
        ):
            bridge_code = self._directed_shared_neighbor_bridge_code()
            vals = np.exp(-np.clip(base_code + bridge_code, 0.0, 60.0)).ravel()
            vals = np.minimum(np.maximum(vals, EPS), 1.0 - EPS)
            directed = coo_matrix((vals, (self._graph_rows_, self._graph_cols_)), shape=(n, n)).tocsr()
            reciprocal = directed.minimum(directed.T)
            reciprocal.setdiag(0.0)
            reciprocal.eliminate_zeros()
            n_comp_reciprocal, _ = connected_components(reciprocal, directed=False)
            completed = directed.maximum(directed.T)
            completed.setdiag(0.0)
            completed.eliminate_zeros()
            n_comp_completed, _ = connected_components(completed, directed=False)
            bridge_active = True

        if self.n_clusters is not None and int(n_comp_reciprocal) == int(self.n_clusters):
            graph = reciprocal
            n_comp = n_comp_reciprocal
            mode = "reciprocal_components"
        else:
            graph = completed
            n_comp = n_comp_completed
            mode = "completed_spectral"

        self.diagnostics_["sample_graph"] = self._graph_diagnostics(graph, k, n_comp)
        self.diagnostics_["sample_graph"].update({
            "mode": mode,
            "reciprocal_components": int(n_comp_reciprocal),
            "completed_components": int(n_comp_completed),
            "shared_neighbor_bridge_active": bool(bridge_active),
        })
        return graph


    def _directed_shared_neighbor_bridge_code(self) -> np.ndarray:
        """Encode local bridge unreliability through shared-neighbor support.

        For a directed local edge i -> j, the bridge support is defined as
        (1 + |N(i) cap N(j)|) / (1 + |N(i)|). The numerator includes the
        transmitted endpoint itself as the minimal support of a valid directed
        neighbor relation, while additional support is provided only by shared
        local neighborhoods. Edges across density valleys or accidental bridges
        usually have fewer shared neighbors and therefore receive a larger
        coding cost. No external threshold or tunable parameter is introduced.
        """
        assert self._knn_ind_ is not None
        ind = self._knn_ind_
        n, k = ind.shape
        if n == 0 or k == 0:
            return np.empty(0, dtype=float)

        marker = np.zeros(n, dtype=np.int32)
        support = np.empty(n * k, dtype=float)
        stamp = 0
        ptr = 0
        denom = float(k + 1)

        for i in range(n):
            stamp += 1
            if stamp >= np.iinfo(np.int32).max:
                marker.fill(0)
                stamp = 1
            neigh_i = ind[i]
            marker[neigh_i] = stamp
            for j in neigh_i:
                shared = int(np.count_nonzero(marker[ind[int(j)]] == stamp))
                support[ptr] = (1.0 + float(shared)) / denom
                ptr += 1

        return -np.log(np.maximum(support, EPS)).reshape(n, k)

    @staticmethod
    def _graph_diagnostics(graph: csr_matrix, graph_order: int, n_components: int) -> Dict:
        return {
            "graph_order": int(graph_order),
            "nnz": int(graph.nnz),
            "components": int(n_components),
            "weight_mean": float(np.mean(graph.data)) if graph.nnz else 0.0,
            "weight_min": float(np.min(graph.data)) if graph.nnz else 0.0,
            "weight_max": float(np.max(graph.data)) if graph.nnz else 0.0,
        }

    # ---------------------------- final partition ----------------------------
    def _estimate_k_by_eigengap(self, graph: csr_matrix) -> int:
        n = graph.shape[0]
        if n <= 2:
            return 1
        max_k = int(min(max(2, math.ceil(math.sqrt(n))), n - 1))
        W = graph.toarray()
        deg = np.sum(W, axis=1)
        D_inv = np.diag(1.0 / np.sqrt(np.maximum(deg, EPS)))
        lap = np.eye(n) - D_inv @ W @ D_inv
        vals = np.sort(np.linalg.eigvalsh(lap)[: max_k + 1])
        gaps = np.diff(vals)
        return 1 if gaps.size <= 1 else int(np.argmax(gaps[1:]) + 2)

    def _partition_graph(self, graph: csr_matrix, k: int) -> np.ndarray:
        n = graph.shape[0]
        if k <= 1:
            self.diagnostics_["partition_mode"] = "single_cluster"
            return np.zeros(n, dtype=int)
        k = int(min(max(k, 1), n))
        n_comp, comp_labels = connected_components(graph, directed=False)
        n_comp = int(n_comp)
        if n_comp == k:
            self.diagnostics_["partition_mode"] = "connected_components"
            return _relabel(comp_labels)
        if 1 < n_comp < k:
            self.diagnostics_["partition_mode"] = "componentwise_spectral"
            return self._componentwise_spectral_partition(graph, comp_labels, n_comp, k)
        self.diagnostics_["partition_mode"] = "global_spectral"
        try:
            labels = SpectralClustering(n_clusters=k, affinity="precomputed", assign_labels="cluster_qr", random_state=0).fit_predict(graph)
        except Exception:
            labels = SpectralClustering(n_clusters=k, affinity="precomputed", assign_labels="discretize", random_state=0).fit_predict(graph)
        return _relabel(labels)

    def _componentwise_spectral_partition(self, graph: csr_matrix, comp_labels: np.ndarray, n_comp: int, k: int) -> np.ndarray:
        """Partition a disconnected graph without mixing independent components.

        When a graph has fewer connected components than the requested cluster
        number, full-graph spectral clustering may be numerically unstable
        because several zero eigenvectors are already determined by the
        components. This routine keeps disconnected components independent and
        allocates the remaining cluster budget according to component sizes,
        then applies spectral clustering only inside the selected components.
        It is a deterministic decomposition of spectral partitioning on a
        disconnected graph rather than a component-level merge/split heuristic.
        """
        n = graph.shape[0]
        sizes = np.bincount(comp_labels, minlength=n_comp).astype(float)
        allocation = np.ones(n_comp, dtype=int)
        remaining = int(k - n_comp)
        for _ in range(max(0, remaining)):
            score = sizes / (allocation.astype(float) + 1.0)
            chosen = int(np.argmax(score))
            allocation[chosen] += 1

        labels = np.full(n, -1, dtype=int)
        offset = 0
        for comp_id, q in enumerate(allocation):
            idx = np.where(comp_labels == comp_id)[0]
            q = int(min(max(q, 1), idx.size))
            if q <= 1 or idx.size <= 1:
                labels[idx] = offset
                offset += 1
                continue
            subgraph = graph[idx][:, idx]
            try:
                sub_labels = SpectralClustering(n_clusters=q, affinity="precomputed", assign_labels="cluster_qr", random_state=0).fit_predict(subgraph)
            except Exception:
                sub_labels = SpectralClustering(n_clusters=q, affinity="precomputed", assign_labels="discretize", random_state=0).fit_predict(subgraph)
            labels[idx] = _relabel(sub_labels) + offset
            offset += q

        if np.any(labels < 0):
            raise RuntimeError("Component-wise partition failed to assign all samples")
        self.diagnostics_["component_allocation"] = [int(v) for v in allocation]
        return _relabel(labels)

    def _labels_to_ball_labels(self, labels: np.ndarray) -> np.ndarray:
        ball_labels = np.zeros(len(self.leaf_nodes_), dtype=int)
        for bi, ball in enumerate(self.leaf_nodes_):
            ball_labels[bi] = _majority(labels[ball.indices])
        return _relabel(ball_labels)

    def _partition_total_code(self, labels: np.ndarray) -> float:
        assert self.X_ is not None and self.graph_ is not None
        labels = _relabel(labels)
        n = self.X_.shape[0]
        K = int(len(np.unique(labels)))
        counts = np.bincount(labels, minlength=K)
        total = math.log(K + 1.0) + n * _entropy_from_counts(counts, total=n)

        for lab in range(K):
            total += self._component_code(np.where(labels == lab)[0])

        coo = self.graph_.tocoo()
        upper = coo.row < coo.col
        for i, j, w in zip(coo.row[upper], coo.col[upper], coo.data[upper]):
            if labels[int(i)] != labels[int(j)]:
                total += -math.log(max(1.0 - float(w), EPS))
        return float(total)

    def _component_code(self, idx: np.ndarray) -> float:
        assert self.X_ is not None and self.graph_ is not None and self.sample_to_ball_ is not None
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0:
            return math.inf

        macro = self._leaf_code(self.X_[idx])[0]
        if idx.size <= 1:
            graph_code = 0.0
        else:
            sub = self.graph_[idx][:, idx].copy().tolil()
            sub.setdiag(0.0)
            sub = sub.tocsr()
            if sub.nnz == 0:
                graph_code = idx.size * math.log(idx.size + 1.0)
            else:
                cost_graph = sub.copy()
                cost_graph.data = -np.log(np.maximum(cost_graph.data, EPS))
                n_comp, _ = connected_components(sub, directed=False)
                mst = minimum_spanning_tree(cost_graph).tocoo()
                graph_code = float(np.sum(mst.data) + max(0, n_comp - 1) * idx.size * math.log(idx.size + 1.0))

        ball_ids = self.sample_to_ball_[idx]
        atom_code = float(np.sum([self.leaf_nodes_[int(bi)].leaf_cost / max(self.leaf_nodes_[int(bi)].n, 1) for bi in ball_ids]))
        return float(min(macro, graph_code + atom_code) + BINARY_MODEL_CODE)

import matplotlib.pyplot as plt
import numpy as np

COLOR_LIST_50 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363",
    "#9ecae1", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
    "#e6ab02", "#a6761d", "#666666", "#a6cee3", "#1f78b4",
    "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f",
    "#ff7f00", "#cab2d6", "#6a3d9a", "#ffff99", "#b15928",
]

def visualize_save(path, h, color, title='', point_size=50, alpha=0.7):
    h = h.astype('float')
    dim = len(h[0])
    z = h

    labels = np.asarray(color).astype(int)
    color_list = COLOR_LIST_50
    n_colors = len(color_list)

    if dim == 2:
        plt.figure(figsize=(10, 10))

        label_max = int(labels.max())
        print(label_max)

        for i, label in enumerate(np.unique(labels)):
            mask = (labels == label)
            if not np.any(mask):
                continue
            plt.scatter(
                z[mask, 0],
                z[mask, 1],
                s=point_size,
                alpha=alpha,
                color=color_list[label % n_colors],
            )

        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.grid(False)
        plt.savefig(path + title + '.pdf', bbox_inches='tight', dpi=60)
        plt.savefig(path + title + '.png', dpi=600, transparent=True, bbox_inches='tight')
        plt.show()

    elif dim >= 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        for i, label in enumerate(np.unique(labels)):
            mask = (labels == label)
            if not np.any(mask):
                continue
            cluster_points = z[mask, :3]
            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                cluster_points[:, 2],
                s=point_size,
                alpha=alpha,
                color=color_list[label % n_colors],
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)

        plt.savefig(path + title + '.pdf', bbox_inches='tight', dpi=600)
        plt.savefig(path + title + '.png', dpi=600, transparent=True, bbox_inches='tight')
        plt.show()
    else:
        print(f"Error")

if __name__ == "__main__":

    road = 'Real'
    fname = 'seeds'
    ftype = 'mat'

    data = loadmat(f"Datasets/{road}/{fname}.{ftype}")
    X = data['Feature']
    y = data['ClassificationResult']

    # with open(f'Datasets/{road}/{fname}.{ftype}', 'r', encoding='utf-8') as f:
    #     data, meta = arff.loadarff(f)
    # df = pd.DataFrame(data).values
    # print(df.shape)
    # X = df[:, :-1].astype(float)
    # y = df[:, -1]

    # data = np.loadtxt(f'Datasets/Synthetic/db2.txt')
    # X = data[:, :-1] # 选择前两列作为特征
    # print(X.shape)
    # y = data[:, -1]  # 选择最后一列作为标签

    # df = pd.read_csv(f'Datasets/Synthetic/N_y.csv',header=None)
    # numpy_array = df.values
    # X = numpy_array[:,1:]
    # y = numpy_array[:,0]
    # print(X.shape, y.shape)

    # data = pd.read_csv('Datasets/Synthetic/banana.csv', header=0)
    # print(data)
    # # 分割特征和标签
    # X = np.array(data.iloc[:, :-1])  # 所有行，去掉第一列（标签列）
    # y = data.iloc[:, -1]

    y = LabelEncoder().fit_transform(y)

    k = len(set(y))
    start = time.time()
    model = MDLGranularBallTreeRegularizedSpectralClustering(n_clusters=k, scale_data=True)
    pred = model.fit_predict(X)
    elapsed = time.time() - start
    rows = []
    rows.append({
        "Mode": "MDL-GBTRSC",
        "K_true": k,
        "K_pred": int(len(np.unique(pred))),
        "TreeNodes": int(len(model.nodes_)),
        "LeafBalls": int(len(model.leaf_nodes_)),
        "GraphOrder": int(model.diagnostics_["sample_graph"]["graph_order"]),
        "ARI": float(adjusted_rand_score(y, pred)),
        "NMI": float(normalized_mutual_info_score(y, pred)),
        "ACC": float(clustering_accuracy(y, pred)),
        "Time(s)": float(elapsed),
        "PartitionCode": float(model.partition_code_) if np.isfinite(model.partition_code_) else None,
    })

    for r in rows:
        print(
            f"K={r['K_pred']} | leaves={r['LeafBalls']:3d} | "
            f"g={r['GraphOrder']:2d} | ARI={r['ARI']:.4f}  ACC={r['ACC']:.4f}   "
            f"NMI={r['NMI']:.4f} | {r['Time(s)']:.4f}s"
        )

    save_path = 'Visualization/'
    visualize_save(save_path, X, pred, title=fname)


