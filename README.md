# MDL-GBTRSC

> **Minimum Description Length based Granular-Ball Tree Regularization for Spectral Clustering**

MDL-GBTRSC is a spectral clustering framework that uses an MDL-induced granular-ball tree to regularize the sample-level affinity graph. Instead of applying spectral clustering directly to a graph defined only by pairwise distances, the method first learns stable local regions under a description-length criterion and then feeds their coding-scale information back into graph construction.

---
## Framework

<p align="center">
  <img src="Framework.png" width="100%" alt="Framework of MDL-GBTRSC">
</p>

<p align="center">
  <em>Figure 1. Framework of MDL-GBTRSC.</em>
</p>

## Overview

Spectral clustering is effective for discovering non-convex structures, but its performance is strongly affected by the affinity graph. Fixed graph construction strategies, such as k-nearest-neighbor graphs and RBF graphs, may be sensitive to neighborhood scale, density variation, noise, and weak local connections.

MDL-GBTRSC connects local granular-ball representation with sample-level graph construction. It builds a graph-continuity-regularized granular-ball tree using the Minimum Description Length principle. The stable leaf balls then provide adaptive coding scales for the final affinity graph.

The resulting graph incorporates three types of information:

- **Sample-level proximity**, captured by local neighborhood affinities.
- **Neighborhood continuity**, preserved through reciprocal preliminary graph regularization.
- **Granular-ball-level coding scale**, introduced by stable leaf balls learned under the MDL criterion.

---

## Main Idea

MDL-GBTRSC follows a simple principle:

> A local split should be accepted only when the reduction in local representation cost is sufficient to compensate for the cost of breaking reliable neighborhood continuity.

For each current granular ball, the method compares two local explanations:

1. **Retain model**  
   The current ball is retained as a stable leaf ball.

2. **Split model**  
   The current ball is split into two child balls, while the reciprocal graph-cut cost penalizes the separation of strongly connected neighboring samples.

The model with the shorter description length is selected. This process is repeated until no leaf ball has a positive MDL gain.

---

## Method Pipeline

### 1. Data Preprocessing and Preliminary Graph Construction

The input data are first normalized into a comparable feature scale. A reciprocal preliminary graph is then constructed from local neighborhood relations. This graph is not directly used as the final clustering graph. Instead, it provides neighborhood-continuity evidence for evaluating candidate granular-ball splits.

### 2. Graph-Continuity-Regularized MDL Tree Construction

Starting from the root ball containing all samples, MDL-GBTRSC grows a granular-ball tree in a best-first manner. For each current leaf ball, the retain model and the split model are compared by their description lengths. A split is accepted only when it gives a positive MDL gain after considering the reciprocal graph-cut cost.

### 3. Tree-Regularized Affinity Construction

After tree construction, each sample belongs to exactly one stable leaf ball. The radius and effective variance of each leaf ball define a local coding scale. These coding scales are incorporated into the edge cost of the final graph, allowing affinities to reflect both pairwise distances and local granular-ball structures.

A shared-neighbor bridge code is further used when the component-count condition is satisfied. This term reduces weak bridge affinities with limited common-neighborhood support without requiring an additional user-specified threshold.

### 4. Graph Selection and Clustering

The final graph is selected from the reciprocal and completed tree-regularized graphs. If the reciprocal graph already contains the specified number of connected components, the component labels are used directly. Otherwise, spectral clustering is performed on the selected graph.

---

## Key Features

- **MDL-based local model selection**  
  Granular-ball generation is formulated as a description-length comparison problem.

- **Graph-continuity regularization**  
  Reciprocal neighborhood relations are used to discourage splits that break reliable local continuity.

- **Adaptive local coding scales**  
  Stable leaf balls provide local scale information for tree-regularized affinity construction.

- **No additional bridge threshold**  
  The shared-neighbor bridge adjustment is activated by a graph component-count condition rather than a manually selected threshold.

---

## Citation

```bibtex
@article{xian2026mdlgbtrsc,
  title   = {Minimum Description Length based Granular-Ball Tree Regularization for Spectral Clustering},
  author  = {Xian, Zeqiang and Liu, Caihui and Zhang, Yong and Qiu, Wenjing},
  journal = {Submitted},
  year    = {2026}
}
```

---

## License

This repository is released for academic research. The license information will be updated after publication.

---

## Related MDL Granular-Ball Works

MDL-GBTRSC is developed as part of an ongoing line of research on **MDL-based granular-ball learning**. Related works include:

1. **MDL-GBG: A Non-parametric and Interpretable Granular-Ball Generation Method for Clustering**  
   Zeqiang Xian, Caihui Liu, Yong Zhang, Wenjing Qiu, Duoqian Miao, Witold Pedrycz  
   <https://doi.org/10.48550/arXiv.2605.08759>

2. **A Boundary-Aware Non-parametric Granular-Ball Classifier Based on Minimum Description Length**  
   Zeqiang Xian, Caihui Liu, Yong Zhang, Wenjing Qiu, Duoqian Miao, Witold Pedrycz  
   <https://doi.org/10.48550/arXiv.2605.11406>

We warmly welcome researchers interested in interpretable, non-parametric, and MDL-based granular-ball learning to follow this research direction and discuss related ideas.

---

## Contact

```text
Zeqiang Xian
Gannan Normal University
Email: xianzeqiang@gnnu.edu.cn
```

```text
Caihui Liu
Gannan Normal University
Email: liucaihui@gnnu.edu.cn
```
