# ChronoLogic: Temporal Image Sequence Ordering

**Live Demo → [priyanka-ganesan-15-temporal-ordering-app-xbzfnv.streamlit.app](https://priyanka-ganesan-15-temporal-ordering-app-xbzfnv.streamlit.app/)**

A vision-language benchmark and algorithm suite for **unsupervised temporal ordering** of image sequences. Given a shuffled set of frames from a real-world process — plant growth, DIY construction, time-lapse — the system recovers the correct chronological order without any supervision signal.

---

## Research Dashboard

The interactive Streamlit dashboard is structured as a research report:

| Section | Description |
|---|---|
| **Abstract** | Project overview, KPIs, problem statement |
| **Dataset** | 7 sequences × 8 frames, category breakdown, sample viewer |
| **Methods: Embedding** | OpenCLIP RN50 pipeline, model details, cached stats |
| **Methods: Similarity** | Cosine similarity matrices, temporal structure scores |
| **Experiments: Ordering** | Algorithm comparison — Random, Greedy NN, Continuity |
| **Results** | Metrics table, Kendall τ, pairwise accuracy charts |
| **Diagnostics** | Error taxonomy, forward/reverse analysis, endpoint distinctiveness, pairwise errors |
| **Ablations & Robustness** | Spectral + Insertion-Sort methods, multi-seed stability, weight sweep |
| **Conclusion** | Key findings, limitations, recommended next milestone |

---

## Key Results

| Method | Pairwise Accuracy | Kendall τ |
|---|---|---|
| Greedy Nearest Neighbor | **61.2%** | **+0.22** |
| Random Baseline | 50.0% | 0.00 |
| Continuity | 29.1% | −0.42 |

---

## Project Structure

```
app.py                   ← Streamlit dashboard (entry point)
embedder.py              ← OpenCLIP frame embedding
evaluate_ordering.py     ← Ordering benchmark runner
requirements.txt         ← Dashboard runtime dependencies
src/chronologic/         ← Core library
  ordering/              ← Ordering algorithms
  evaluation/            ← Metrics & runner
  analysis/              ← Diagnostics modules
scripts/
  run_experiments.py     ← Ablation & robustness experiments
  run_diagnostics.py     ← Diagnostic plot generation
Data/
  embeddings/openclip/   ← Cached .npy embeddings
  analysis/              ← All results CSVs
  manifests/sequences.json
```

---

## Run Locally

```bash
# 1. Clone the repo and create a virtual environment
git clone <your-repo-url>
cd Temporal-Ordering
python -m venv .venv && source .venv/bin/activate

# 2. Install dashboard dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```

To re-generate embeddings or run experiments:
```bash
pip install open_clip_torch torch matplotlib  # heavy packages, not in default requirements
python embedder.py
python scripts/run_experiments.py
python scripts/run_diagnostics.py
```

---

## Embedding Model

- **Architecture**: OpenCLIP RN50 (openai weights)
- **Embedding dimension**: 1024
- **Similarity metric**: Cosine
- **Fine-tuned**: No — frozen weights throughout

---

## Citation / Reference

If you use this work, please cite the project repository and note the OpenCLIP model used:

> Ganesan, P. (2026). *ChronoLogic: Unsupervised Temporal Ordering of Image Sequences*. GitHub.
