"""
GraphMem Evaluation Module
==========================

Benchmarks using MultiHopRAG dataset from HuggingFace:
https://huggingface.co/datasets/yixuantt/MultiHopRAG

Dataset:
- 2556 QA samples (multi-hop queries)
- 609 corpus documents (news articles)

Run evaluation:
    python -m graphmem.evaluation.run_eval
"""

__all__ = ["run_eval"]
