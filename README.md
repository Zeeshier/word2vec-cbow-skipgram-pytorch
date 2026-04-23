# Word2Vec in PyTorch: CBOW and Skip-gram

This repository provides a reproducible Word2Vec pipeline in PyTorch with:

- Continuous Bag-of-Words (CBOW)
- Skip-gram
- Nearest-neighbor and analogy evaluation
- Automated paper artifact generation (figures, tables, logs)
- IEEE manuscript assets for reporting

## Author

- Name: Zeeshan Ahmad
- Registration No: 70169515
- University: University of Lahore, Sargodha Campus
- Email: 70169515@student.uol.edu.pk

## Repository Structure

```text
word2vec_pytorch_project/
|
|- data/
|  |- corpus.txt
|
|- paper/
|  |- ieee_word2vec_zeeshan.tex
|  |- manuscript_template.md
|  |- references.bib
|
|- results/
|  |- checkpoints/
|  |- embeddings/
|  |- figures/
|  |- paper/
|     |- figures/
|     |- tables/
|     |- logs/
|     |- README.md
|
|- src/
|  |- preprocess.py
|  |- dataset.py
|  |- models.py
|  |- train.py
|  |- evaluate.py
|  |- generate_paper_artifacts.py
|  |- utils.py
|
|- requirements.txt
|- README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Use a plain text corpus at:

```text
data/corpus.txt
```

One sentence per line is recommended.

## Train Models

Train CBOW:

```bash
python src/train.py \
  --data_path data/corpus.txt \
  --model_type cbow \
  --embedding_dim 100 \
  --window_size 2 \
  --min_count 1 \
  --batch_size 256 \
  --epochs 5 \
  --lr 0.05
```

Train Skip-gram:

```bash
python src/train.py \
  --data_path data/corpus.txt \
  --model_type skipgram \
  --embedding_dim 100 \
  --window_size 2 \
  --min_count 1 \
  --batch_size 256 \
  --epochs 5 \
  --lr 0.05
```

Checkpoint naming pattern:

```text
best_{model_type}_d{embedding_dim}_w{window_size}.pt
```

## Evaluate

```bash
python src/evaluate.py \
  --checkpoint_path results/checkpoints/best_skipgram_d100_w2.pt \
  --query_word king \
  --analogy "man king woman"
```

Analogy format:

```text
A B C  ->  A : B :: C : ?
```

## Generate Paper Artifacts

Run:

```bash
python src/generate_paper_artifacts.py
```

Generated outputs:

- results/paper/figures: loss comparison, ablation, embedding PCA plots
- results/paper/tables: dataset stats, main results, ablations, qualitative outputs
- results/paper/logs: per-run logs and environment snapshot
- results/paper/README.md: artifact index

## Research Notes

- Training currently uses full softmax with cross-entropy.
- This is suitable for controlled experiments and implementation studies.
- For larger-scale reproduction, extend with negative sampling or hierarchical softmax.


