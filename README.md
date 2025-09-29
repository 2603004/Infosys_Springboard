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

## Observations & Next steps
- The models produce different similarity distributions; MPNet and MiniLM often cluster similar algorithmic patterns together.
- For deeper code understanding, swap in code-pretrained models such as `microsoft/codebert-base` or `Salesforce/codet5-*`. These require tokenization changes and potentially different pooling strategies.
- Consider fine-tuning or using dedicated code embedding models for larger datasets.

## Notes
- This pipeline is intentionally minimal and reproducible in Colab.
- Extend snippet list to scale analysis. When scaling to hundreds/thousands of snippets, batch encoding and GPU runtime are recommended.

