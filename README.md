# Milestone 1 — Code Explainer

## Overview
This milestone implements a pipeline that:
- Collects ≥10 code snippets.
- Parses each snippet using Python's `ast` to extract functions, classes, imports and detect code patterns (recursion, decorators, list comprehensions, async).
- Tokenizes code using Python's `tokenize` module.
- Combines AST-derived summaries with raw code to produce textual representations for embedding.
- Encodes representations using three pretrained embedding models: **MiniLM**, **DistilRoBERTa**, and **MPNet** (from `sentence-transformers`).
- Computes pairwise cosine similarities, visualizes similarity heatmaps, and performs PCA + t-SNE for embedding comparison.
- Compares model outputs (similarities and visual clusters).

## How to run
1. Open Google Colab.
2. Create a new notebook and paste the ordered cells from the provided notebook.
3. Run the cells in order. The notebook installs dependencies, loads models, computes embeddings, saves outputs in `/outputs`.

## Files
- `Milestone1/code_explainer_notebook.ipynb` — the Colab notebook (this repo contains the `.ipynb`).
- `Milestone1/README.md` — this document.
- `Milestone1/outputs/` — CSVs and numpy embedding files produced by the notebook (created when you run it).

## Methodology
1. **AST Parsing**: `ast.parse()` followed by `ast.walk()` to discover definitions and patterns.
2. **Tokenization**: Python `tokenize` gives token-level detail; optional HF tokenizers can be used for more model-aligned tokenization.
3. **Textualization**: Construct a short textual description including AST metadata and code body, so sentence embedding models get both semantics and structure.
4. **Embedding Models**: `sentence-transformers` models used:
   - `all-MiniLM-L6-v2` (MiniLM)
   - `all-distilroberta-v1` (DistilRoBERTa)
   - `all-mpnet-base-v2` (MPNet)
5. **Comparison**: Cosine similarity matrices, pairwise scatterplots comparing model similarities, and PCA+t-SNE visualizations.

📊 Observations and Model Comparison

| Model                                    | Embedding Dimension | Key Characteristics                                           | Performance Observation                                                                                      |
| ---------------------------------------- | ------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **MiniLM (all-MiniLM-L6-v2)**            | 384                 | Lightweight, fast, optimized for sentence-level meaning       | Captures structural and functional similarity well. Factorial and Fibonacci cluster closely.                 |
| **DistilRoBERTa (all-distilroberta-v1)** | 768                 | Deeper transformer with robust language understanding         | Captures textual semantics better than structure; sometimes confuses variable naming similarities.           |
| **MPNet (all-mpnet-base-v2)**            | 768                 | Advanced model combining masked and permuted token prediction | Achieves highest semantic consistency; closely clusters algorithmically similar snippets (recursion, loops). |

📈 Visual Results

Heatmaps: Show dense high-similarity zones for functionally related snippets.

PCA/t-SNE Plots: Reveal tighter and more semantically consistent clusters for MPNet and MiniLM than DistilRoBERTa.

Distribution: MPNet produces more uniform embeddings, whereas DistilRoBERTa shows scattered variance.

## Notes
- This pipeline is intentionally minimal and reproducible in Colab.
- Extend snippet list to scale analysis. When scaling to hundreds/thousands of snippets, batch encoding and GPU runtime are recommended.

