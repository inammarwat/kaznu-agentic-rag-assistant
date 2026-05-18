# KazNU Agentic RAG Assistant

This repository contains the implementation of an Agentic Retrieval-Augmented Generation system for university information assistance at Al-Farabi Kazakh National University.

## Project scope

The system processes university-related PDFs and web pages, builds a vector database, and compares a Baseline RAG pipeline with an Agentic RAG pipeline.

## Main components

- PDF and web data ingestion
- Text cleaning and preprocessing
- Table extraction and tuition-fee normalization
- Chunking and metadata enrichment
- Chroma vector store creation
- Baseline RAG question answering
- Agentic RAG with:
  - query decomposition
  - multi-query retrieval
  - reciprocal-rank fusion
  - source validation
  - source sufficiency scoring
  - reflection agent
- LLM-as-a-judge evaluation
- Baseline vs Agentic RAG comparison
- Evaluation graphs for thesis and presentation

## Experimental status

The stable Qwen/KazNU-based experiment includes:

- Baseline RAG on 50 simple queries
- Baseline RAG on 30 complex queries
- Agentic RAG on the same 30 complex queries
- Evaluation metrics: faithfulness, relevance, completeness, citation quality, hallucination score, and latency
- PNG plots for presentation

## Security note

API keys are loaded from `.env`, which is intentionally excluded from Git tracking.
Use `.env.example` as a template.