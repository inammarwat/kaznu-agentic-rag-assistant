from graphviz import Digraph
from pathlib import Path

output_dir = Path("outputs/thesis_figures")
output_dir.mkdir(parents=True, exist_ok=True)

dot = Digraph("KazNU_Agentic_RAG_Methodology", format="png")
dot.attr(rankdir="TB", splines="ortho", nodesep="0.4", ranksep="0.6")
dot.attr(label="Methodology Architecture of the Proposed Agentic RAG System", labelloc="t", fontsize="18")

# Global node style
dot.attr("node", shape="box", style="rounded,filled", fillcolor="lightgoldenrodyellow", color="black", fontsize="11")

# Data sources
dot.node("A1", "PDF Sources\n(Academic Policy, Booklet,\nTuition Fee, AI Regulation)")
dot.node("A2", "Web URL Sources\n(29 university webpages)")

# Ingestion
dot.node("B", "Data Ingestion Pipeline\nPDF text extraction\nWeb content extraction\nTable extraction")

# Preprocessing
dot.node("C", "Preprocessing and Cleaning\ntext cleaning\nwhitespace normalization\nnoise removal")

# Structured tuition
dot.node("D", "Tuition Table Normalization\nfaculty\napplicant region\ndegree level\nlanguage\nacademic year\nfee")

# Chunking
dot.node("E", "Chunking and Metadata Enrichment\nPDF chunks\nWeb chunks\nTable chunks\nStructured fact chunks")

# Embeddings
dot.node("F", "Embedding Generation\nBAAI/bge-small-en-v1.5")

# Vector DB
dot.node("G", "Vector Database\nChromaDB")

# Baseline RAG
dot.node("H", "Baseline RAG\nsingle-query retrieval\ntop-k search\nLLM answer generation")

# Agentic RAG
dot.node("I", "Agentic RAG\nquery decomposition\nmulti-query retrieval\nrank fusion\nsource validation")

# Reflection
dot.node("J", "P1 Extension\nsource sufficiency scoring\nreflection / critique\nanswer revision")

# Adaptive routing
dot.node("K", "Adaptive Routing\nsimple query → Baseline RAG\ncomplex query → Agentic RAG")

# Evaluation
dot.node("L", "Evaluation Framework\nLLM-as-a-judge\nhuman evaluation template\ncomparison reports\nPNG figures")

# Outputs
dot.node("M", "Final Outputs\nanswers with citations\nmetrics\nreports\ngraphs")

# Edges
dot.edge("A1", "B")
dot.edge("A2", "B")
dot.edge("B", "C")
dot.edge("B", "D")
dot.edge("C", "E")
dot.edge("D", "E")
dot.edge("E", "F")
dot.edge("F", "G")
dot.edge("G", "H")
dot.edge("G", "I")
dot.edge("I", "J")
dot.edge("H", "K")
dot.edge("J", "K")
dot.edge("K", "L")
dot.edge("L", "M")

# Render
output_path = dot.render(filename="figure_4_methodology_architecture", directory=str(output_dir), cleanup=True)
print("Saved:", output_path)