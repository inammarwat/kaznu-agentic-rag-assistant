# Baseline RAG vs Agentic RAG — Evaluation Comparison

## Methodological Note

Both systems were evaluated on the same 30-question complex-query set.

This provides a controlled head-to-head comparison between Baseline RAG and Agentic RAG under the same query conditions. The comparison should be interpreted not only through answer-quality scores, but also through latency, transparency, decomposition behavior, source-validation behavior, and hallucination control.

---

## Overall Comparison

| system | questions_evaluated | avg_faithfulness_score | avg_answer_relevance_score | avg_context_relevance_score | avg_completeness_score | avg_citation_quality_score | avg_hallucination_score | hallucination_rate | avg_latency_seconds | avg_subqueries_per_question | avg_validated_sources_per_question | avg_rejected_sources_per_question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline RAG | 30 | 4.967 | 5 | 5 | 5 | 5 | 5 | 0.0 | 1.797 | N/A | N/A | N/A |
| Agentic RAG | 30 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | 3.2 | 3.467 | 4.833 | 2.033 |

---

## Category-Level Comparison

| system | category | count | avg_faithfulness_score | avg_answer_relevance_score | avg_context_relevance_score | avg_completeness_score | avg_citation_quality_score | avg_hallucination_score | hallucination_rate | avg_latency_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline RAG | academic_policy | 6 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | None |
| Baseline RAG | admissions | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | None |
| Baseline RAG | ai_policy | 6 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | None |
| Baseline RAG | missing_information | 1 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | None |
| Baseline RAG | mixed_policy | 1 | 4 | 5 | 5 | 5 | 5 | 5 | 0.0 | None |
| Baseline RAG | noisy_retrieval | 2 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | None |
| Baseline RAG | tuition_admissions | 6 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | None |
| Baseline RAG | university_info | 3 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | None |
| Agentic RAG | academic_policy | 6 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | 3.296 |
| Agentic RAG | admissions | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | 3.775 |
| Agentic RAG | ai_policy | 6 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | 3.409 |
| Agentic RAG | missing_information | 1 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | 2.084 |
| Agentic RAG | mixed_policy | 1 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | 5.048 |
| Agentic RAG | noisy_retrieval | 2 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | 2.676 |
| Agentic RAG | tuition_admissions | 6 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | 2.2 |
| Agentic RAG | university_info | 3 | 5 | 5 | 5 | 5 | 5 | 5 | 0.0 | 3.734 |

---

## Thesis Interpretation

The Baseline RAG system provides a strong and efficient retrieval-generation pipeline for university information assistance. The Agentic RAG system extends this pipeline with query decomposition, multi-query retrieval, source validation, and validated-context answer generation.

On the 30-question complex-query set, both systems achieved zero hallucination, while Agentic RAG achieved slightly higher faithfulness. The main contribution of Agentic RAG should therefore be interpreted as increased transparency, control, and source-level validation rather than only raw score improvement.

Agentic RAG provides an explicit trace of subqueries, retrieved sources, rejected sources, and validated sources. This trace supports explainability and hallucination control for complex university information queries. The tradeoff is increased latency because each query involves additional reasoning and validation steps.
