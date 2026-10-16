# FinRAG — Financial Knowledge RAG Agent

A fully local Retrieval-Augmented Generation pipeline over financial explainers, SEC filings (10-K / 10-Q) and arXiv market-microstructure papers, answered by a local LLM (Ollama) — with portfolio-aware queries over your own holdings and automatic citation checking.

> **Privacy by design:** no cloud APIs. Embeddings, vector store and generation all run on this machine; portfolio data never leaves it.

**Full documentation: [finrag/README.md](finrag/README.md)** — progress, architecture, corpus, setup, known issues and roadmap.

```bash
cd finrag
pip install -r requirements.txt && ollama pull llama3.2
python scripts/rag_loop.py --interactive
```
