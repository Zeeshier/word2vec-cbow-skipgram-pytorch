# Word2Vec (CBOW + Skip-gram) Implementation Study

## Abstract

Write 150-250 words covering:
- problem and motivation
- method (CBOW/Skip-gram in PyTorch)
- key results
- conclusion

## 1. Introduction

- Problem statement
- Why word embeddings matter
- Contributions of this work

## 2. Related Work

- Mikolov et al. Word2Vec
- CBOW vs Skip-gram differences
- Efficiency methods (hierarchical softmax/negative sampling)

## 3. Methodology

### 3.1 Data Pipeline

Use details from:
- `src/preprocess.py`
- `src/dataset.py`

### 3.2 Model Architectures

Use details from:
- `src/models.py`

Include equations for CBOW and Skip-gram objectives.

### 3.3 Training Procedure

Use details from:
- `src/train.py`
- `results/paper/tables/hyperparameters_main.csv`

## 4. Experimental Setup

### 4.1 Dataset

Use:
- `results/paper/tables/dataset_statistics.csv`

### 4.2 Environment and Reproducibility

Use:
- `requirements.txt`
- `results/paper/logs/environment.txt`

### 4.3 Evaluation Protocol

Use:
- `src/evaluate.py`
- `results/paper/tables/nearest_neighbors.csv`
- `results/paper/tables/analogy_results.csv`

## 5. Results and Discussion

### 5.1 Main Comparison

Use:
- `results/paper/tables/main_results_cbow_vs_skipgram.csv`
- `results/paper/figures/cbow_vs_skipgram_loss.png`

### 5.2 Ablation Study

Use:
- `results/paper/tables/ablation_window_embeddingdim.csv`
- `results/paper/figures/ablation_window_embeddingdim.png`

### 5.3 Qualitative Analysis

Use:
- `results/paper/tables/nearest_neighbors.csv`
- `results/paper/tables/analogy_results.csv`
- `results/paper/figures/embedding_pca_skipgram.png`
- `results/paper/figures/embedding_pca_cbow.png`

## 6. Error Analysis

Document:
- weak analogies
- unstable neighbors
- effects of small corpus size

## 7. Limitations

Discuss:
- small synthetic corpus
- full softmax computational limits
- no negative sampling/hierarchical softmax

## 8. Conclusion and Future Work

Summarize findings and propose next improvements.

## Appendix A. Reproduction Commands

Add exact commands used for training, artifact generation, and evaluation.
