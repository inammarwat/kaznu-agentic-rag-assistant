# Baseline RAG vs Agentic RAG — Evaluation Comparison

## Methodological Note

The two systems were evaluated on different query sets:

- **Baseline RAG** was evaluated on 50 dataset-aligned questions.
- **Agentic RAG** was evaluated on 15 complex multi-part questions.

Therefore, this table should be interpreted as a comparison of system behavior under different query difficulty settings, not as a strict same-question head-to-head comparison.

For a fully controlled comparison, run both Baseline RAG and Agentic RAG on the same complex-question set.

---

## Overall Comparison

| system | questions_evaluated | avg_faithfulness_score | avg_answer_relevance_score | avg_context_relevance_score | avg_completeness_score | avg_citation_quality_score | avg_hallucination_score | hallucination_rate | avg_latency_seconds | avg_subqueries_per_question | avg_validated_sources_per_question | avg_rejected_sources_per_question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline RAG | 50 | 4.94 | 5 | 5 | 5 | 4.92 | 4.96 | 0.04 | 2.193 | N/A | N/A | N/A |
| Agentic RAG | 15 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | 3.144 | 3.667 | 5.067 | 1.467 |

---

## Category-Level Comparison

| system | category | count | avg_faithfulness_score | avg_answer_relevance_score | avg_context_relevance_score | avg_completeness_score | avg_citation_quality_score | avg_hallucination_score | hallucination_rate | avg_latency_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline RAG | academic_policy | 15 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | None |
| Baseline RAG | admissions | 10 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | None |
| Baseline RAG | ai_policy | 10 | 4.8 | 5 | 5 | 5 | 4.8 | 4.9 | 0.1 | None |
| Baseline RAG | tuition | 10 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | None |
| Baseline RAG | university_info | 5 | 4.8 | 5 | 5 | 5 | 4.6 | 4.8 | 0.2 | None |
| Agentic RAG | academic_policy | 4 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | 3.28 |
| Agentic RAG | admissions | 2 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | 3.639 |
| Agentic RAG | ai_policy | 3 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | 3.677 |
| Agentic RAG | mixed_policy | 1 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | 4.213 |
| Agentic RAG | tuition_admissions | 3 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | 1.865 |
| Agentic RAG | university_info | 2 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | 2.963 |

---

## Thesis Interpretation

The Baseline RAG system demonstrated strong performance on direct and dataset-aligned queries with lower average latency. The Agentic RAG system introduced query decomposition, multi-query retrieval, source validation, and validated-context generation for complex multi-part queries. Its higher latency is expected because each question involves additional reasoning and validation steps.

The main contribution of the Agentic RAG design is not only score improvement, but the explicit reasoning trace: decomposition, subqueries, retrieved sources, rejected sources, and validated sources. This trace improves transparency and supports hallucination control in complex university information assistance.
