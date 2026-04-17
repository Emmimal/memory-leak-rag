# memory-leak-rag
RAG gets more confident as it gets more wrong. Three measurable failure modes + a managed memory architecture that fixes them. Companion code for Towards Data Science.

# memory-leak-rag

> A pure-Python simulation of RAG memory failure modes — and the managed memory architecture that fixes them.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-1.0.0-green)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

Most RAG systems monitor confidence. Confidence looks fine — even trends upward.
Memory accumulates. Accuracy collapses. This is the failure nobody measures, demonstrated
in four phases with real benchmark numbers and a working fix.

Read the full write-up on Towards Data Science → **[RAG Gets Worse as It Grows — I Built the Memory Layer That Stops It](#)**
*(https://towardsdatascience.com/)*

---

## What It Does

```
Memory Pool → Unbounded Agent → Relevance Decay → Confidence Divergence → Wrong Answer
                                                                               ↓
                                                                       Managed Agent
                                                                               ↓
                  Topic Routing → Deduplication → Relevance Eviction → Correct Answer
```

Four failure modes. Four architectural fixes. One Python file:

| Phase | What It Shows |
|-------|---------------|
| Phase 1 — Relevance Decay | Retrieved context collapses from 44% relevant to 14% as memory grows |
| Phase 2 — Confidence Divergence | Accuracy drops 20 points while confidence rises 7.6 points |
| Phase 3 — Stale Dominance | A VPN certificate entry outscores a password-reset entry by 0.014 cosine similarity |
| Phase 4 — Managed Memory | Four mechanisms restore accuracy — same queries, same agent |

---

## Installation

```bash
git clone https://github.com/Emmimal/memory-leak-rag.git
cd memory-leak-rag
pip install numpy scipy colorama
```

No other dependencies. No API key. No model downloads. No GPU.
Embeddings are deterministic and keyword-seeded — same text produces the same
vector every run without any model required.

---

## Quick Start

```bash
# Run the full four-phase demo
python llm_memory_leak_demo.py

# Suppress INFO logs
python llm_memory_leak_demo.py --quiet

# Disable coloured terminal output
python llm_memory_leak_demo.py --no-color

# Run unit tests first (recommended — verifies correctness logic)
python llm_memory_leak_demo.py --test
```

All results reproduce in under 10 seconds on CPU.

---

## The Core Finding

```
Memory Size    Accuracy    Avg Confidence
──────────────────────────────────────────
10 entries      50%          70.4%
100 entries     50%          74.7%
200 entries     40%          75.8%
500 entries     30%          78.0%
```

**Accuracy dropped 20 percentage points. Confidence rose 7.6 points.
Any monitoring system that alerts on low confidence will never fire.**

Phase 3 makes this concrete — one query, one wrong answer, exact similarity scores:

```
LARGE MEMORY (200 entries) — silently broken

[1] ✗ sim=0.471  turn=158  VPN certificate expires in 30 days notify users
[2] ✓ sim=0.457  turn=  2  POST /auth/reset resets user password via email

Answer:  VPN certificate expires in 30 days notify users
Correct: False  |  Confidence: 78.5%
```

The VPN entry wins by a margin of **0.014**. That invisible gap is the entire
difference between a correct answer and a wrong one.

---

## The Fix: Managed Memory Architecture

Four mechanisms, all load-bearing. Remove any one and accuracy degrades:

| Mechanism | What It Closes |
|-----------|----------------|
| **Query-intent topic routing** | Cross-topic stale entries never enter the candidate set |
| **Semantic deduplication at ingestion** | Repeated stale content cannot accumulate retrieval weight |
| **Relevance-scored eviction** | Oldest correct answers survive over newest off-topic noise |
| **Lexical reranking (BM25-inspired)** | Same-topic wrong entries separated by token overlap bonus |

**Result at 200-entry input:**

| Metric | Unbounded (200) | Managed (50 retained) |
|--------|-----------------|----------------------|
| Relevance rate | 22% | 42% |
| Accuracy | 40% | **60%** |
| Avg confidence | 75.8% | 77.5% |
| Memory footprint | 200 entries | **50 entries** |

Same query, now correct:

```
MANAGED MEMORY (200 entries ingested, 50 retained)

[1] ✓ sim=0.608  turn=  2  POST /auth/reset resets user password via email

Answer:  POST /auth/reset resets user password via email
Correct: True  |  Confidence: 80.0%
```

One-quarter of the memory. Twenty percentage points more accurate.
**The constraint is the feature.**

---

## Project Structure

```
memory-leak-rag/
└── llm_memory_leak_demo.py     # Complete four-phase demo + unit tests
```

The entire implementation is a single self-contained Python file.

**Internal structure:**

| Class / Function | Job |
|------------------|-----|
| `UnboundedMemoryAgent` | Naive agent — cosine similarity only, no eviction |
| `ManagedMemoryAgent` | Four-mechanism agent — routing, dedup, eviction, reranking |
| `_generate_memory_pool()` | Synthetic conversation generator with growing stale ratio |
| `_measure_at_size()` | Measurement harness — runs all queries, returns stats |
| `TestAnswerKeywords` | Unit tests — verifies correctness logic closes topic-level loophole |

---

## Performance (CPU only, 10-query benchmark)

| Operation | Latency |
|-----------|---------|
| Memory pool generation (500 entries) | < 1 ms |
| Unbounded agent — full benchmark | < 50 ms |
| Managed agent — full benchmark | < 50 ms |
| Full four-phase demo | < 10 s |

No embedding model is loaded at runtime. All latency is pure Python + numpy.

---

## When to Use This

**Worth reading if you have:**
- A RAG system that accumulates retrieved documents over time
- An AI copilot or agent with a persistent memory store
- A customer support agent that logs past interactions
- Any LLM workflow where context grows across sessions

**Skip it if you have:**
- Single-turn queries against a small fixed knowledge base
- A fully deterministic domain where keyword retrieval is sufficient and auditable
- No memory accumulation — the failure modes here require context growth to emerge

---

## Known Limitations

- Embeddings are keyword-seeded deterministic vectors, not learned sentence encoders.
  Behaviour in high-dimensional real embedding spaces may differ at the margins.
- Topic clusters are hand-labeled. Production systems require automated cluster discovery.
- Confidence is modelled as a linear function of mean retrieval score — real systems
  use calibrated probabilities that behave differently under distribution shift.
- These simplifications make failure modes easier to observe, not harder. The structural
  causes persist regardless of embedding dimension or model scale.

---

## Related

This repository is part of a series on building real LLM systems:

| Article | Repository |
|---------|-----------|
| RAG Isn't Enough — I Built the Missing Context Layer | [context-engine](https://github.com/Emmimal/context-engine) |
| RAG Gets Worse as It Grows — I Built the Memory Layer That Stops It | **this repo** |

---

## References

[1] Robertson & Zaragoza (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval.* https://doi.org/10.1561/1500000019

[2] Yi Luan, Jacob Eisenstein, Kristina Toutanova, Michael Collins; Sparse, Dense, and Attentional Representations for Text Retrieval. Transactions of the Association for Computational Linguistics 2021; 9 329–345. *TACL.* doi: https://doi.org/10.1162/tacl_a_00369

[3] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang; Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Association for Computational Linguistics 2024; 12 157–173. *TACL.* https://doi.org/10.1162/tacl_a_00638

[4] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS.* https://arxiv.org/abs/2005.11401

[5] Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan. (2023). Precise zero-shot dense retrieval without relevance labels. *ACL 2023.* https://arxiv.org/abs/2212.10496

---

## License

MIT
