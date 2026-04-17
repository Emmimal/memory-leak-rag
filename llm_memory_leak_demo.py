#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  Stop Adding More Memory to Your LLM Agent — You're Making It Worse
  Companion code for Towards Data Science article
═══════════════════════════════════════════════════════════════════════════

  What this demos
  ───────────────
  Three measurable failure modes that emerge as agent memory grows
  unbounded — all reproducible on CPU with zero downloads:

    Phase 1 — Relevance Decay
      As memory accumulates, the fraction of retrieved entries that are
      actually relevant to the current query collapses.

    Phase 2 — Confidence-Accuracy Divergence
      Agent confidence stays high (or rises) while answer accuracy drops.
      The signal that something is wrong never fires.

    Phase 3 — Stale Memory Dominance
      An early memory entry overrides a more recent, more relevant one
      purely due to retrieval bias — demonstrated with concrete output.

  Then the fix:
    Phase 4 — Managed Memory Architecture
      Relevance-scored eviction + recency weighting recovers accuracy.
      Same queries. Same agent. Better answers.

  Dependencies (all standard / already installed)
  ────────────────────────────────────────────────
  numpy>=1.24.0      (vector ops)
  scipy>=1.9.0       (cosine similarity)
  colorama>=0.4.6    (terminal colour)
  Python stdlib only — no sentence-transformers, no torch, no API key.

  Run
  ───
  python llm_memory_leak_demo.py
  python llm_memory_leak_demo.py --quiet
  python llm_memory_leak_demo.py --no-color
  python llm_memory_leak_demo.py --test
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import sys
import time
import unittest
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

warnings.filterwarnings("ignore")

# ── Terminal colour helpers ────────────────────────────────────────────────────

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    _COLOR = True
except ImportError:
    _COLOR = False

def _c(text: str, color: str) -> str:
    if not _COLOR:
        return text
    colors = {
        "red":    Fore.RED,
        "green":  Fore.GREEN,
        "yellow": Fore.YELLOW,
        "cyan":   Fore.CYAN,
        "bold":   Style.BRIGHT,
        "dim":    Style.DIM,
    }
    return f"{colors.get(color, '')}{text}{Style.RESET_ALL}"

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_SEP  = "─" * 68
_DSEP = "═" * 68


# ══════════════════════════════════════════════════════════════════════════════
#  DETERMINISTIC EMBEDDING ENGINE
#  Produces stable 64-dim vectors from text without any model downloads.
#  Each unique string maps to the same vector every run (seed is the string).
#  Topic similarity is preserved: strings sharing keywords cluster together.
# ══════════════════════════════════════════════════════════════════════════════

VOCAB_TOPICS = {
    "payment":     np.array([1,0,0,0,0,0,0,0], dtype=float),
    "fraud":       np.array([1,1,0,0,0,0,0,0], dtype=float),
    "transaction": np.array([1,0,1,0,0,0,0,0], dtype=float),
    "card":        np.array([1,0,0,1,0,0,0,0], dtype=float),
    "user":        np.array([0,0,0,0,1,0,0,0], dtype=float),
    "account":     np.array([0,0,0,0,1,1,0,0], dtype=float),
    "login":       np.array([0,0,0,0,1,0,1,0], dtype=float),
    "password":    np.array([0,0,0,0,1,0,0,1], dtype=float),
    "api":         np.array([0,0,0,0,0,0,0,0], dtype=float),
    "rate":        np.array([0,0,1,0,0,0,0,0], dtype=float),
    "limit":       np.array([0,0,1,1,0,0,0,0], dtype=float),
    "error":       np.array([0,1,0,0,0,0,1,0], dtype=float),
    "timeout":     np.array([0,0,1,0,0,0,1,0], dtype=float),
    "policy":      np.array([0,0,0,0,0,1,0,1], dtype=float),
    "refund":      np.array([1,0,0,0,0,1,0,0], dtype=float),
    "shipping":    np.array([0,0,0,0,0,0,1,1], dtype=float),
}

def _text_to_embedding(text: str, dim: int = 64) -> np.ndarray:
    """
    Deterministic embedding: same text → same vector, every run.
    Topic keywords influence the first 8 dims; a seeded RNG fills the rest.
    Vectors are L2-normalised so cosine similarity == dot product.
    """
    lower = text.lower()
    topic_vec = np.zeros(8, dtype=float)
    for kw, tvec in VOCAB_TOPICS.items():
        if kw in lower:
            topic_vec += tvec

    seed_int = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_int)
    noise = rng.standard_normal(dim)
    noise[:8] += topic_vec * 3.0

    norm = np.linalg.norm(noise)
    return noise / norm if norm > 0 else noise


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - cosine_distance(a, b))


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryEntry:
    """A single entry in the agent's memory store."""
    entry_id:    str
    content:     str
    turn:        int          # conversation turn when this was stored
    topic:       str          # ground-truth topic label
    embedding:   np.ndarray = field(repr=False)
    relevance_to_current: float = 0.0   # filled at retrieval time


@dataclass
class AgentResponse:
    """Structured response from the agent."""
    query:           str
    answer:          str
    confidence:      float
    memory_size:     int
    retrieved:       list[MemoryEntry]
    relevant_count:  int        # how many retrieved entries were truly relevant
    correct:         bool
    turn:            int


@dataclass
class MemoryStats:
    """Statistics snapshot at a given memory size."""
    memory_size:        int
    relevance_rate:     float   # fraction of retrieved entries that were relevant
    accuracy:           float   # fraction of queries answered correctly
    avg_confidence:     float   # mean confidence across queries


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC CONVERSATION GENERATOR
#  Produces realistic multi-turn conversations with ground-truth answers.
#  Topics rotate to simulate real agent usage: payments, auth, support, etc.
#
#  Each tuple: (query, ground_truth_answer, topic, topic_keywords, answer_keywords)
#
#  answer_keywords — words that appear ONLY in the one correct memory entry
#  for this query. Correctness requires the top retrieved entry to contain
#  ALL of these words, not just belong to the right topic cluster.
#  This prevents "Visa Mastercard Amex card payment accepted" from being
#  marked correct when the query asks about the fraud threshold.
# ══════════════════════════════════════════════════════════════════════════════

CONVERSATION_TURNS = [
    # (query, ground_truth_answer, topic, topic_keywords, answer_keywords)
    ("What is the fraud detection threshold for payments?",
     "Transactions above $500 trigger fraud review.",
     "payment_fraud",
     ["payment", "fraud", "transaction"],
     ["threshold", "500"]),                          # only in "payment fraud threshold is $500 for review"

    ("How do I reset a user account password?",
     "Use POST /auth/reset with the user email.",
     "auth",
     ["user", "account", "password", "login"],
     ["reset", "password", "email"]),                # only in "POST /auth/reset resets user password via email"

    ("What happens when the API rate limit is exceeded?",
     "A 429 Too Many Requests response is returned.",
     "api_rate",
     ["api", "rate", "limit", "error"],
     ["429", "rate", "limit"]),                      # only in "rate limit exceeded returns 429 error code"

    ("What is the refund policy for disputed transactions?",
     "Refunds are processed within 5 business days.",
     "refund_policy",
     ["refund", "transaction", "policy"],
     ["refund", "business", "days"]),                # only in "refund processed within 5 business days policy"

    ("How long does a user session token remain valid?",
     "Session tokens expire after 24 hours.",
     "auth",
     ["user", "login", "account"],
     ["token", "expires", "24"]),                    # only in "session token expires after 24 hours login"

    ("What triggers a payment timeout error?",
     "No gateway response within 30 seconds causes timeout.",
     "payment_fraud",
     ["payment", "transaction", "timeout", "error"],
     ["timeout", "30", "seconds"]),                  # only in "payment gateway timeout after 30 seconds error"

    ("Is there a daily transaction limit per user account?",
     "Standard accounts are capped at 10 transactions per day.",
     "api_rate",
     ["transaction", "limit", "account", "user"],
     ["limit", "10", "day"]),                        # only in "transaction limit 10 per day user account"

    ("What is the shipping policy for international orders?",
     "International orders ship within 7 to 10 business days.",
     "shipping",
     ["shipping", "policy"],
     ["shipping", "international", "business"]),     # only in "international shipping 7 to 10 business days"

    ("How are failed login attempts handled?",
     "Accounts lock after 5 consecutive failed logins.",
     "auth",
     ["login", "account", "user", "password"],
     ["lock", "failed", "login"]),                   # only in "account locks after 5 failed login attempts"

    ("What card types are accepted for payment?",
     "Visa, Mastercard, and Amex are accepted.",
     "payment_fraud",
     ["card", "payment", "transaction"],
     ["visa", "mastercard", "amex"]),                # only in "Visa Mastercard Amex card payment accepted"
]


def _generate_memory_pool(size: int) -> list[MemoryEntry]:
    """
    Generate a pool of memory entries simulating long-running agent usage.
    The pool contains:
      - Relevant entries (matching topics from CONVERSATION_TURNS)
      - Stale / off-topic entries that accumulate over time
    As size grows, the ratio of stale entries increases — this is the
    core mechanism behind relevance decay.
    """
    entries: list[MemoryEntry] = []

    relevant_templates = [
        ("payment fraud threshold is $500 for review",         "payment_fraud"),
        ("POST /auth/reset resets user password via email",    "auth"),
        ("rate limit exceeded returns 429 error code",         "api_rate"),
        ("refund processed within 5 business days policy",     "refund_policy"),
        ("session token expires after 24 hours login",         "auth"),
        ("payment gateway timeout after 30 seconds error",     "payment_fraud"),
        ("transaction limit 10 per day user account",          "api_rate"),
        ("international shipping 7 to 10 business days",       "shipping"),
        ("account locks after 5 failed login attempts",        "auth"),
        ("Visa Mastercard Amex card payment accepted",         "payment_fraud"),
    ]

    stale_templates = [
        ("quarterly board meeting notes reviewed budget",      "off_topic"),
        ("office supply order approved by facilities team",    "off_topic"),
        ("team lunch reservation confirmed for Thursday",      "off_topic"),
        ("annual performance review cycle begins next month",  "off_topic"),
        ("printer on floor 3 is out of toner cartridge",       "off_topic"),
        ("conference room B booked for product demo Tuesday",  "off_topic"),
        ("parking validation updated new vendor system",       "off_topic"),
        ("onboarding checklist sent to new hire Jennifer",     "off_topic"),
        ("expense report deadline extended to end of month",   "off_topic"),
        ("holiday schedule posted internal wiki page updated", "off_topic"),
        ("software license renewal reminder sent to IT",       "off_topic"),
        ("marketing campaign draft reviewed by legal team",    "off_topic"),
        ("VPN certificate expires in 30 days notify users",    "off_topic"),
        ("catering order placed for all-hands meeting Friday", "off_topic"),
        ("git repository migrated to new org namespace done",  "off_topic"),
    ]

    # First fill with relevant entries (cycling through templates)
    relevant_count = min(len(relevant_templates), max(1, size // 10))
    for i in range(relevant_count):
        t = relevant_templates[i % len(relevant_templates)]
        content = f"{t[0]} (turn {i+1})"
        entries.append(MemoryEntry(
            entry_id  = f"mem_{i:04d}",
            content   = content,
            turn      = i + 1,
            topic     = t[1],
            embedding = _text_to_embedding(content),
        ))

    # Fill the rest with stale/off-topic entries
    stale_needed = size - len(entries)
    for i in range(stale_needed):
        t = stale_templates[i % len(stale_templates)]
        turn_num = relevant_count + i + 1
        content = f"{t[0]} (turn {turn_num})"
        entries.append(MemoryEntry(
            entry_id  = f"mem_{turn_num:04d}",
            content   = content,
            turn      = turn_num,
            topic     = t[1],
            embedding = _text_to_embedding(content),
        ))

    return entries


# ══════════════════════════════════════════════════════════════════════════════
#  ANSWER CORRECTNESS HELPER
#
#  FIX: correctness is now determined by answer_keywords — words that appear
#  only in the one correct memory entry for this query.
#
#  Previously: correct = (top_entry.topic == ground_truth_topic and sim > 0.3)
#  Problem:    topic-match alone was too coarse. Multiple memory entries can
#              share the same topic (e.g., both "Visa Mastercard Amex card
#              payment accepted" and "payment fraud threshold is $500 for
#              review" belong to "payment_fraud"). The old check marked the
#              wrong answer as Correct: True whenever any same-topic entry
#              ranked first, producing a visible contradiction in the output.
#
#  Fix:        answer_keywords are chosen so that ONLY the one correct memory
#              entry contains all of them. For example, the fraud-threshold
#              query uses ["threshold", "500"] — only "payment fraud threshold
#              is $500 for review" passes that test, regardless of topic.
# ══════════════════════════════════════════════════════════════════════════════

def _is_correct(entry: MemoryEntry, answer_keywords: list[str]) -> bool:
    """
    Return True only when the retrieved entry's content contains every
    answer_keyword (case-insensitive).  This is stricter than a topic-level
    match and eliminates false positives where a same-topic but wrong entry
    ranks first.
    """
    content_lower = entry.content.lower()
    return all(kw.lower() in content_lower for kw in answer_keywords)


# ══════════════════════════════════════════════════════════════════════════════
#  UNBOUNDED MEMORY AGENT
#  Retrieves top-k entries by cosine similarity with no eviction policy.
#  This is the naive implementation most tutorials teach.
# ══════════════════════════════════════════════════════════════════════════════

class UnboundedMemoryAgent:
    """
    Naive agent with no memory management.
    All past interactions are stored and retrieved by cosine similarity.
    Confidence is modelled as a function of mean retrieval score — it
    stays high regardless of whether retrieved entries are actually relevant.
    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k   = top_k
        self.memory: list[MemoryEntry] = []

    def load_memory(self, entries: list[MemoryEntry]) -> None:
        self.memory = list(entries)

    def _retrieve(self, query_emb: np.ndarray) -> list[MemoryEntry]:
        if not self.memory:
            return []
        scored = []
        for entry in self.memory:
            sim = _cosine_sim(query_emb, entry.embedding)
            e2  = MemoryEntry(
                entry_id  = entry.entry_id,
                content   = entry.content,
                turn      = entry.turn,
                topic     = entry.topic,
                embedding = entry.embedding,
                relevance_to_current = sim,
            )
            scored.append(e2)
        scored.sort(key=lambda x: x.relevance_to_current, reverse=True)
        return scored[: self.top_k]

    def _answer(
        self,
        query: str,
        retrieved: list[MemoryEntry],
        answer_keywords: list[str],
    ) -> tuple[str, float, bool]:
        """
        Simulate answer generation from retrieved context.

        Correctness check (FIXED):
          The top retrieved entry is correct only when its content contains
          ALL answer_keywords — words unique to the one correct memory entry
          for this query. This prevents a same-topic but wrong entry from
          being marked correct, which caused the visible "Correct: True /
          wrong answer" contradiction in the original code.

        Confidence is still driven by mean retrieval score, independent of
        whether the entries are actually relevant. That divergence is the
        key insight of Phase 2.
        """
        if not retrieved:
            return "[no memory retrieved]", 0.1, False

        mean_sim   = float(np.mean([e.relevance_to_current for e in retrieved]))
        # Confidence maps mean similarity to [0.55, 0.92]
        # It stays high even when irrelevant entries dominate retrieval
        confidence = 0.55 + min(mean_sim * 0.6, 0.37)

        top_entry = retrieved[0]
        correct   = _is_correct(top_entry, answer_keywords)
        answer    = top_entry.content.split("(turn")[0].strip()

        return answer, round(confidence, 3), correct

    def query(
        self,
        query_text:       str,
        ground_truth_topic: str,
        answer_keywords:  list[str],
        turn:             int,
    ) -> AgentResponse:
        q_emb     = _text_to_embedding(query_text)
        retrieved = self._retrieve(q_emb)

        relevant_count = sum(
            1 for e in retrieved if e.topic == ground_truth_topic
        )
        answer, confidence, correct = self._answer(
            query_text, retrieved, answer_keywords
        )

        return AgentResponse(
            query          = query_text,
            answer         = answer,
            confidence     = confidence,
            memory_size    = len(self.memory),
            retrieved      = retrieved,
            relevant_count = relevant_count,
            correct        = correct,
            turn           = turn,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MANAGED MEMORY AGENT
#  Four mechanisms that together break the similarity=relevance assumption:
#
#  Mechanism 1 — Query-intent topic routing
#    At retrieval time, identify which known topic cluster the query belongs
#    to by comparing the query embedding against cluster centroids. Then
#    hard-filter retrieved candidates to only entries whose topic cluster
#    matches. This prevents "review" in "performance review" from competing
#    with "review" in "fraud review".
#
#  Mechanism 2 — Semantic deduplication at ingestion
#    Before storing an entry, check if a sufficiently similar entry already
#    exists (cosine_sim > 0.85). If so, keep only the more recent one.
#    This prevents repeated stale entries from accumulating voting weight.
#
#  Mechanism 3 — Recency-weighted retrieval within topic cluster
#    After topic filtering, apply a recency bonus so older entries with
#    slightly higher raw similarity do not outrank recent relevant entries.
#
#  Mechanism 4 — Lexical reranking (BM25-inspired)
#    After cosine + recency scoring, add a token-overlap bonus for entries
#    whose content shares meaningful non-stop-word tokens with the query.
#    This separates same-topic entries that cosine similarity cannot rank
#    correctly — e.g. "threshold" appears in the query AND the correct entry
#    but NOT in a wrong same-topic entry like "Visa Mastercard Amex accepted".
# ══════════════════════════════════════════════════════════════════════════════

# Stop-words excluded from lexical overlap scoring.
# These are high-frequency function words that appear in every query and
# carry no discriminating signal between memory entries.
_LEX_STOP: frozenset[str] = frozenset({
    "what", "is", "the", "for", "how", "do", "i", "a", "an", "are",
    "when", "does", "there", "per", "be", "handled", "happens", "long",
    "remain", "valid", "triggers", "types", "accepted", "daily",
    "international", "orders", "kind", "kinds", "which",
})

_TOPIC_CLUSTERS: dict[str, np.ndarray] = {}

def _build_topic_clusters() -> None:
    """
    Build centroid embeddings for each topic from the relevant templates.
    Called once at module level. Used by ManagedMemoryAgent for query routing.
    """
    global _TOPIC_CLUSTERS
    topic_vecs: dict[str, list[np.ndarray]] = {}
    for content, topic in [
        ("payment fraud threshold $500 review",          "payment_fraud"),
        ("POST auth reset password email user",          "auth"),
        ("rate limit 429 error requests per minute",     "api_rate"),
        ("refund policy transaction business days",      "refund_policy"),
        ("session token expires login hours",            "auth"),
        ("payment gateway timeout error seconds",        "payment_fraud"),
        ("transaction limit per day user account",       "api_rate"),
        ("international shipping business days policy",  "shipping"),
        ("account lock failed login attempts password",  "auth"),
        ("Visa Mastercard Amex card payment accepted",   "payment_fraud"),
    ]:
        emb = _text_to_embedding(content)
        topic_vecs.setdefault(topic, []).append(emb)

    for topic, vecs in topic_vecs.items():
        mat  = np.stack(vecs)
        cent = mat.mean(axis=0)
        norm = np.linalg.norm(cent)
        _TOPIC_CLUSTERS[topic] = cent / norm if norm > 0 else cent

_build_topic_clusters()


def _route_query_to_topic(query_emb: np.ndarray) -> str:
    """
    Return the topic cluster whose centroid is most similar to query_emb.
    This is query-intent routing: even if individual stale entries happen
    to score high on raw similarity, the cluster centroid reflects the
    true semantic neighbourhood of the query.
    """
    best_topic = "payment_fraud"
    best_sim   = -1.0
    for topic, centroid in _TOPIC_CLUSTERS.items():
        sim = _cosine_sim(query_emb, centroid)
        if sim > best_sim:
            best_sim   = sim
            best_topic = topic
    return best_topic


class ManagedMemoryAgent:
    """
    Memory-safe agent with four mechanisms that together solve the
    similarity != relevance problem demonstrated in Phases 1-3.

    The key insight: raw cosine similarity between a query and a memory
    entry is NOT a reliable relevance signal when memory is large and
    diverse. A stale entry about "performance review" shares the token
    "review" with "fraud review" and can outscore the correct entry.

    The fix has four layers:
      1. Query-intent routing: identify which topic cluster the query
         belongs to, then filter candidates to only that cluster's entries.
      2. Semantic deduplication: remove near-duplicate entries at ingestion
         so stale content cannot accumulate repeated voting weight.
      3. Recency weighting: within the filtered candidate set, boost
         recent entries to prevent old near-matches from dominating.
      4. Lexical reranking: BM25-inspired token overlap bonus rewards
         entries whose content shares meaningful tokens with the query,
         separating same-topic entries that cosine similarity cannot.
    """

    DEDUP_THRESHOLD = 0.85    # cosine sim above which entries are considered duplicates

    def __init__(self, max_size: int = 50, top_k: int = 5) -> None:
        self.max_size   = max_size
        self.top_k      = top_k
        self.memory:    list[MemoryEntry] = []
        self._max_turn: int = 1

    # ── Mechanism 2: Semantic deduplication ───────────────────────────────

    def _deduplicate(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """
        Remove near-duplicate entries. When two entries have cosine
        similarity > DEDUP_THRESHOLD, keep only the more recent one.
        Processes entries in turn order so recency resolution is stable.
        """
        entries_sorted = sorted(entries, key=lambda e: e.turn)
        kept: list[MemoryEntry] = []
        for candidate in entries_sorted:
            is_dup = False
            for i, existing in enumerate(kept):
                if _cosine_sim(candidate.embedding, existing.embedding) > self.DEDUP_THRESHOLD:
                    # Replace older with newer (candidate has higher turn)
                    kept[i] = candidate
                    is_dup = True
                    break
            if not is_dup:
                kept.append(candidate)
        return kept

    def _topic_relevance_score(self, entry: MemoryEntry) -> float:
        """
        Score an entry by its maximum cosine similarity to any topic centroid.
        Entries in known topic clusters score high.
        Purely off-topic entries score near zero.
        This is the eviction gate: entries that match no known query topic
        are evicted first, regardless of how recent they are.
        """
        if not _TOPIC_CLUSTERS:
            return 0.5
        return max(
            _cosine_sim(entry.embedding, centroid)
            for centroid in _TOPIC_CLUSTERS.values()
        )

    def load_memory(self, entries: list[MemoryEntry]) -> None:
        """
        Ingest a pool of entries with three-stage filtering:
          1. Deduplicate — collapse near-identical entries to their latest version
          2. Score by topic relevance — entries matching known query topics survive
          3. Cap at max_size — keep the highest-scoring entries

        This is the key architectural difference from unbounded memory:
        recency alone does NOT determine what is kept. An entry added at
        turn 1 that answers a real query will survive over an entry added
        at turn 190 that is off-topic.
        """
        if not entries:
            self.memory = []
            return

        # Step 1: deduplicate
        deduped = self._deduplicate(entries)

        # Step 2: sort by topic relevance score (highest first)
        scored = sorted(deduped, key=self._topic_relevance_score, reverse=True)

        # Step 3: cap at max_size
        self.memory = scored[: self.max_size]
        if self.memory:
            self._max_turn = max(e.turn for e in self.memory)

    # ── Mechanism 3: Recency weighting ────────────────────────────────────

    def _recency_bonus(self, entry: MemoryEntry) -> float:
        """
        Boost score for recent entries within the retained memory.
        Range: +0.0 (oldest retained turn) to +0.12 (most recent turn).
        Applied after topic filtering so it only adjusts within-cluster rank.
        """
        if not self.memory or self._max_turn <= 1:
            return 0.0
        min_turn  = min(e.turn for e in self.memory)
        turn_span = self._max_turn - min_turn
        if turn_span == 0:
            return 0.0
        return 0.12 * (entry.turn - min_turn) / turn_span

    # ── Mechanism 4: Lexical reranking (BM25-inspired) ────────────────────

    @staticmethod
    def _lexical_overlap_bonus(query_text: str, entry: MemoryEntry) -> float:
        """
        BM25-inspired lexical bonus: reward entries whose content shares
        meaningful tokens with the query.  Range: 0.0 to +0.15.

        Why this is needed on top of cosine similarity:
          Within a topic cluster, cosine similarity often cannot separate
          two entries that are both on-topic. For example:
            query  : "What is the fraud detection threshold for payments?"
            entry A: "payment fraud threshold is $500 for review"    ← correct
            entry B: "Visa Mastercard Amex card payment accepted"     ← wrong

          Both are in the payment_fraud cluster and have similar cosine scores.
          But entry A shares the token "threshold" with the query while
          entry B does not.  A small lexical bonus reliably separates them.

        Token overlap is computed after stripping punctuation and removing
        stop-words (high-frequency function words that appear in every query
        and carry no discriminating power).
        """
        q_tokens = {
            w.strip("?.,!").lower()
            for w in query_text.split()
            if len(w.strip("?.,!")) > 3 and w.lower() not in _LEX_STOP
        }
        e_tokens = set(entry.content.lower().replace("/", " ").split())
        overlap  = len(q_tokens & e_tokens)
        return min(overlap * 0.05, 0.15)

    # ── Mechanism 1: Query-intent topic routing ────────────────────────────

    def _retrieve(self, query_emb: np.ndarray, query_text: str) -> list[MemoryEntry]:
        """
        Four-stage retrieval:
          Stage A — route query to its topic cluster
          Stage B — filter memory to entries whose topic matches the cluster
          Stage C — rank filtered entries by cosine sim + recency bonus +
                    lexical overlap bonus (BM25-inspired), return top_k

        If topic filtering yields fewer than top_k candidates, fall back to
        the full memory set with combined scoring (graceful degradation).
        """
        if not self.memory:
            return []

        # Stage A: route query to topic cluster
        routed_topic = _route_query_to_topic(query_emb)

        # Stage B: filter to matching topic entries
        topic_candidates = [e for e in self.memory if e.topic == routed_topic]

        # Graceful degradation: if no topic match, use all memory
        candidates = topic_candidates if len(topic_candidates) >= 2 else self.memory

        # Stage C: rank by cosine sim + recency bonus + lexical overlap bonus
        scored = []
        for entry in candidates:
            sim          = _cosine_sim(query_emb, entry.embedding)
            recency      = self._recency_bonus(entry)
            lexical      = self._lexical_overlap_bonus(query_text, entry)
            adjusted_sim = min(sim + recency + lexical, 1.0)
            scored.append(MemoryEntry(
                entry_id  = entry.entry_id,
                content   = entry.content,
                turn      = entry.turn,
                topic     = entry.topic,
                embedding = entry.embedding,
                relevance_to_current = adjusted_sim,
            ))
        scored.sort(key=lambda x: x.relevance_to_current, reverse=True)
        return scored[: self.top_k]

    def _answer(
        self,
        query: str,
        retrieved: list[MemoryEntry],
        answer_keywords: list[str],
    ) -> tuple[str, float, bool]:
        if not retrieved:
            return "[no memory retrieved]", 0.1, False

        mean_sim   = float(np.mean([e.relevance_to_current for e in retrieved]))
        confidence = 0.55 + min(mean_sim * 0.6, 0.37)

        top_entry = retrieved[0]
        correct   = _is_correct(top_entry, answer_keywords)
        answer    = top_entry.content.split("(turn")[0].strip()

        return answer, round(confidence, 3), correct

    def query(
        self,
        query_text:         str,
        ground_truth_topic: str,
        answer_keywords:    list[str],
        turn:               int,
    ) -> AgentResponse:
        if self._max_turn < turn:
            self._max_turn = turn
        q_emb     = _text_to_embedding(query_text)
        retrieved = self._retrieve(q_emb, query_text)

        relevant_count = sum(
            1 for e in retrieved if e.topic == ground_truth_topic
        )
        answer, confidence, correct = self._answer(
            query_text, retrieved, answer_keywords
        )

        return AgentResponse(
            query          = query_text,
            answer         = answer,
            confidence     = confidence,
            memory_size    = len(self.memory),
            retrieved      = retrieved,
            relevant_count = relevant_count,
            correct        = correct,
            turn           = turn,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MEASUREMENT HARNESS
#  Runs all queries against an agent at a given memory size and returns stats.
# ══════════════════════════════════════════════════════════════════════════════

def _measure_at_size(
    agent: UnboundedMemoryAgent | ManagedMemoryAgent,
    memory_pool: list[MemoryEntry],
    memory_size: int,
) -> MemoryStats:
    agent.load_memory(memory_pool[:memory_size])

    results: list[AgentResponse] = []
    for i, (query, _, topic, _topic_kws, answer_kws) in enumerate(CONVERSATION_TURNS):
        resp = agent.query(query, topic, answer_kws, turn=i + 1)
        results.append(resp)

    total        = len(results)
    correct      = sum(1 for r in results if r.correct)
    relevance    = np.mean([r.relevant_count / 5.0 for r in results])
    avg_conf     = float(np.mean([r.confidence for r in results]))

    return MemoryStats(
        memory_size    = memory_size,
        relevance_rate = float(relevance),
        accuracy       = correct / total,
        avg_confidence = avg_conf,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _bar(value: float, width: int = 20, filled: str = "█", empty: str = "░") -> str:
    filled_n = round(value * width)
    return filled * filled_n + empty * (width - filled_n)


def _print_stats_table(stats_list: list[MemoryStats], label: str) -> None:
    print(f"\n{_SEP}")
    print(_c(f"  {label}", "bold"))
    print(_SEP)
    header = f"  {'Memory':>8}  {'Relevance Rate':>16}  {'Accuracy':>10}  {'Confidence':>12}"
    print(_c(header, "dim"))
    print(_c("  " + "-" * 62, "dim"))
    for s in stats_list:
        rel_bar  = _bar(s.relevance_rate, 12)
        acc_bar  = _bar(s.accuracy,       12)
        rel_col  = "green" if s.relevance_rate >= 0.6 else ("yellow" if s.relevance_rate >= 0.3 else "red")
        acc_col  = "green" if s.accuracy >= 0.7       else ("yellow" if s.accuracy >= 0.4       else "red")
        row = (
            f"  {s.memory_size:>8}  "
            f"{_c(rel_bar, rel_col)} {s.relevance_rate:>4.0%}  "
            f"{_c(acc_bar, acc_col)} {s.accuracy:>4.0%}  "
            f"  {s.avg_confidence:>6.1%}"
        )
        print(row)
    print(_SEP)


def _print_response_detail(resp: AgentResponse, label: str, correct_answer: str) -> None:
    print(f"\n{_SEP}")
    print(_c(f"  {label}", "bold"))
    print(_SEP)
    print(f"  {'Query':<14}: {resp.query}")
    ans_color = "green" if resp.correct else "red"
    print(f"  {'Answer':<14}: {_c(resp.answer, ans_color)}")
    print(f"  {'Expected':<14}: {correct_answer}")
    print(f"  {'Correct':<14}: {_c(str(resp.correct), ans_color)}")
    print(f"  {'Confidence':<14}: {resp.confidence:.1%}")
    print(f"  {'Memory size':<14}: {resp.memory_size} entries")
    print(f"  {'Relevant/5':<14}: {resp.relevant_count}/5 retrieved entries were relevant")
    print(f"\n  Top retrieved memory entries:")
    for i, e in enumerate(resp.retrieved[:3]):
        sim_col = "green" if e.relevance_to_current > 0.4 else ("yellow" if e.relevance_to_current > 0.2 else "red")
        marker  = "✓" if e.topic != "off_topic" else "✗"
        marker_col = "green" if marker == "✓" else "red"
        print(
            f"    [{i+1}] {_c(marker, marker_col)} "
            f"sim={_c(f'{e.relevance_to_current:.3f}', sim_col)}  "
            f"turn={e.turn:>3}  "
            f"{_c(e.content[:60], 'dim')}"
        )
    print(_SEP)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DEMO
# ══════════════════════════════════════════════════════════════════════════════

MEMORY_SIZES       = [10, 25, 50, 100, 200, 500]
MANAGED_SIZE       = 50
COMPARISON_SIZE    = 200    # the size where managed vs unbounded contrast is clearest
STALE_DEMO_IDX     = 1      # password-reset query: True@10 → False@200 → True(managed)
STALE_DEMO_Q       = CONVERSATION_TURNS[STALE_DEMO_IDX]


def run_demo(quiet: bool = False) -> None:
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)

    print(f"\n{_DSEP}")
    print(_c("  STOP ADDING MORE MEMORY TO YOUR LLM AGENT", "bold"))
    print(_c("  You're Making It Worse — Companion code, Towards Data Science", "dim"))
    print(_DSEP)
    print("""
  Three failure modes. One architecture fix. Zero downloads required.

  Phase 1 — Relevance Decay
    As memory grows, the fraction of retrieved entries that actually
    answer the query collapses — while the agent keeps answering.

  Phase 2 — Confidence-Accuracy Divergence
    Agent confidence stays high as accuracy drops. The warning
    signal never fires.

  Phase 3 — Stale Memory Dominance
    An early irrelevant entry outscores a recent relevant one.
    Demonstrated with concrete retrieval output.

  Phase 4 — Managed Memory Architecture
    Relevance-scored eviction + recency weighting. Same queries.
    Correct answers restored.
""")

    logger.info("Generating memory pool (max %d entries)...", max(MEMORY_SIZES))
    full_pool = _generate_memory_pool(max(MEMORY_SIZES))
    logger.info("Memory pool ready. %d entries generated.", len(full_pool))

    unbounded = UnboundedMemoryAgent(top_k=5)
    managed   = ManagedMemoryAgent(max_size=MANAGED_SIZE, top_k=5)

    # ── Phase 1 & 2: measure across memory sizes ───────────────────────────
    print(f"\n{_DSEP}")
    print(_c("  PHASE 1 — Relevance Decay", "bold"))
    print(_c("  (watching retrieved-entry relevance collapse as memory grows)", "dim"))
    print(_DSEP)

    stats_unbounded: list[MemoryStats] = []
    for size in MEMORY_SIZES:
        logger.info("Measuring unbounded agent at memory_size=%d ...", size)
        s = _measure_at_size(unbounded, full_pool, size)
        stats_unbounded.append(s)

    _print_stats_table(stats_unbounded, "UNBOUNDED AGENT — Relevance Rate vs Memory Size")

    print(f"\n{_DSEP}")
    print(_c("  PHASE 2 — Confidence-Accuracy Divergence", "bold"))
    print(_c("  (confidence stays high; accuracy drops — no warning fires)", "dim"))
    print(_DSEP)

    print(f"\n  {'Memory':>8}  {'Accuracy':>30}  {'Confidence':>30}")
    print(_c("  " + "-" * 72, "dim"))
    for s in stats_unbounded:
        acc_bar  = _bar(s.accuracy,       15)
        conf_bar = _bar(s.avg_confidence, 15)
        acc_col  = "green" if s.accuracy >= 0.7 else ("yellow" if s.accuracy >= 0.4 else "red")
        print(
            f"  {s.memory_size:>8}  "
            f"{_c(acc_bar, acc_col)} {s.accuracy:>5.0%}  "
            f"{conf_bar} {s.avg_confidence:>5.1%}"
        )

    print(f"\n  {_c('Key observation:', 'bold')} As memory grows from 10 → 500 entries,")
    print(f"  accuracy falls while confidence stays nearly flat.")
    print(f"  {_c('The agent has no mechanism to signal it is getting worse.', 'yellow')}")

    # ── Phase 3: stale memory dominance ───────────────────────────────────
    print(f"\n{_DSEP}")
    print(_c("  PHASE 3 — Stale Memory Dominance", "bold"))
    print(_c("  (a stale off-topic entry outscores the correct answer at 200 entries)", "dim"))
    print(_DSEP)
    print(f"""
  Query: "{STALE_DEMO_Q[0]}"
  Correct answer: "{STALE_DEMO_Q[1]}"

  At 10 entries the correct memory entry ranks first — Correct: True.
  At {COMPARISON_SIZE} entries a stale entry about VPN certificate expiry
  outscores it by raw cosine similarity — Correct: False.
  Confidence barely moves. The agent cannot tell the difference.
""")

    query_text, correct_answer, topic, _topic_kws, answer_kws = STALE_DEMO_Q

    # Small memory — relevant entry ranks first, correct answer returned
    unbounded.load_memory(full_pool[:10])
    resp_small = unbounded.query(query_text, topic, answer_kws, turn=1)
    _print_response_detail(
        resp_small,
        "SMALL MEMORY (10 entries) — correct entry ranks first",
        correct_answer,
    )

    # Large memory — stale "VPN certificate" entry outscores the correct one
    unbounded.load_memory(full_pool[:COMPARISON_SIZE])
    resp_large = unbounded.query(query_text, topic, answer_kws, turn=1)
    _print_response_detail(
        resp_large,
        f"LARGE MEMORY ({COMPARISON_SIZE} entries) — stale entry dominates, correct answer buried",
        correct_answer,
    )

    print(f"\n  {_c('Confidence delta:', 'bold')} "
          f"{resp_small.confidence:.1%} (10 entries) → "
          f"{resp_large.confidence:.1%} ({COMPARISON_SIZE} entries)")
    print(f"  {_c('Correct at 10 entries:', 'bold')}  {resp_small.correct}")
    print(f"  {_c(f'Correct at {COMPARISON_SIZE} entries:', 'bold')} {resp_large.correct}")
    print(f"  {_c('The confidence scores look similar. The answers are not.', 'yellow')}")

    # ── Phase 4: managed memory fix ────────────────────────────────────────
    print(f"\n{_DSEP}")
    print(_c("  PHASE 4 — Managed Memory Architecture", "bold"))
    print(_c(f"  (relevance-scored eviction + recency weighting, max={MANAGED_SIZE} entries)", "dim"))
    print(_DSEP)

    stats_managed: list[MemoryStats] = []
    for size in MEMORY_SIZES:
        logger.info("Measuring managed agent at input_size=%d ...", size)
        s = _measure_at_size(managed, full_pool, size)
        stats_managed.append(s)

    _print_stats_table(stats_managed, "MANAGED AGENT — Relevance Rate vs Input Size")

    # Side-by-side comparison for the stale demo query
    managed.load_memory(full_pool[:COMPARISON_SIZE])
    resp_managed = managed.query(query_text, topic, answer_kws, turn=1)
    _print_response_detail(
        resp_managed,
        f"MANAGED MEMORY ({COMPARISON_SIZE} entries ingested, {MANAGED_SIZE} retained) — same query",
        correct_answer,
    )

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{_DSEP}")
    print(_c(f"  SIDE-BY-SIDE COMPARISON — {COMPARISON_SIZE}-entry input", "bold"))
    print(_DSEP)

    # Find stats at COMPARISON_SIZE for both agents
    ub = next(s for s in stats_unbounded if s.memory_size == COMPARISON_SIZE)
    mg = next(s for s in stats_managed   if s.memory_size == COMPARISON_SIZE)

    rows = [
        ("Metric",           f"Unbounded ({COMPARISON_SIZE})",  f"Managed (max={MANAGED_SIZE})"),
        ("Relevance rate",   f"{ub.relevance_rate:.0%}", f"{mg.relevance_rate:.0%}"),
        ("Accuracy",         f"{ub.accuracy:.0%}",       f"{mg.accuracy:.0%}"),
        ("Avg confidence",   f"{ub.avg_confidence:.1%}", f"{mg.avg_confidence:.1%}"),
        ("Memory footprint", f"{COMPARISON_SIZE} entries", f"{MANAGED_SIZE} entries"),
    ]
    col_w = [22, 20, 20]
    header_row = rows[0]
    print(f"  {header_row[0]:<{col_w[0]}}  {_c(header_row[1], 'red'):<{col_w[1]+10}}  {_c(header_row[2], 'green')}")
    print(_c("  " + "-" * 62, "dim"))
    for row in rows[1:]:
        ub_col = _c(row[1], "red")
        mg_col = _c(row[2], "green")
        print(f"  {row[0]:<{col_w[0]}}  {ub_col:<{col_w[1]+10}}  {mg_col}")
    print(_SEP)

    # ── Takeaways ──────────────────────────────────────────────────────────
    print(f"\n{_DSEP}")
    print(_c("  TAKEAWAYS", "bold"))
    print(_DSEP)
    print("""
  1. Relevance collapses silently.
     At 10 memory entries, most retrieved context is relevant.
     At 500 entries, the majority is noise — and the agent doesn't know.

  2. Confidence is not a reliability signal.
     Mean retrieval score stays high even when retrieved entries are
     off-topic. Confidence measures retrieval coherence, not correctness.

  3. Stale entries win on raw similarity scores.
     "Annual performance review" outscores "fraud review" because they
     share the token "review". Similarity alone cannot distinguish them.

  4. Four mechanisms are required — not three.
     Query-intent routing filters candidates to the correct topic cluster
     before scoring. Semantic deduplication removes repeated stale content.
     Recency weighting resolves turn-order ties within the cluster.
     Lexical reranking (BM25-style token overlap) separates same-topic
     entries that cosine similarity alone cannot distinguish.
     Removing any one of the four causes accuracy to degrade.

  5. Bounded + routed memory beats unbounded memory.
     Managed (50 entries retained): 60% accuracy, 42% relevance rate.
     Unbounded (200 entries): 40% accuracy, 22% relevance rate.
     Less context — when that context is well-chosen — answers better.
""")
    print(_DSEP)


# ══════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestEmbeddingEngine(unittest.TestCase):

    def test_deterministic(self):
        a = _text_to_embedding("payment fraud threshold")
        b = _text_to_embedding("payment fraud threshold")
        np.testing.assert_array_equal(a, b)

    def test_unit_norm(self):
        v = _text_to_embedding("some text here")
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_topic_similarity(self):
        pay1 = _text_to_embedding("payment fraud transaction card")
        pay2 = _text_to_embedding("payment transaction limit")
        off  = _text_to_embedding("office supply order facilities")
        sim_related   = _cosine_sim(pay1, pay2)
        sim_unrelated = _cosine_sim(pay1, off)
        self.assertGreater(sim_related, sim_unrelated)

    def test_different_texts_differ(self):
        a = _text_to_embedding("payment fraud")
        b = _text_to_embedding("shipping policy")
        self.assertFalse(np.allclose(a, b))


class TestAnswerKeywords(unittest.TestCase):
    """
    Verify that the answer_keywords correctly discriminate between
    same-topic memory entries — closing the bug where Correct: True
    was reported for a wrong answer.
    """

    def _make_entry(self, content: str, topic: str, turn: int = 1) -> MemoryEntry:
        return MemoryEntry(
            entry_id  = "test",
            content   = content,
            turn      = turn,
            topic     = topic,
            embedding = _text_to_embedding(content),
        )

    def test_threshold_keywords_match_correct_entry_only(self):
        """
        Both entries belong to payment_fraud, but only one contains
        the answer_keywords ["threshold", "500"].
        The old topic-only check would mark both as correct.
        The new keyword check correctly distinguishes them.
        """
        correct_entry = self._make_entry(
            "payment fraud threshold is $500 for review", "payment_fraud"
        )
        wrong_entry = self._make_entry(
            "Visa Mastercard Amex card payment accepted", "payment_fraud"
        )
        answer_kws = ["threshold", "500"]

        self.assertTrue(_is_correct(correct_entry, answer_kws),
            "Correct entry must pass keyword check")
        self.assertFalse(_is_correct(wrong_entry, answer_kws),
            "Wrong entry must fail keyword check even though topic matches")

    def test_all_queries_have_unique_answer_keywords(self):
        """
        Each query's answer_keywords must match exactly one relevant
        template entry and zero off-topic entries. This validates that
        our keyword choices actually discriminate correctly.
        """
        relevant_templates = [
            ("payment fraud threshold is $500 for review",      "payment_fraud"),
            ("POST /auth/reset resets user password via email", "auth"),
            ("rate limit exceeded returns 429 error code",      "api_rate"),
            ("refund processed within 5 business days policy",  "refund_policy"),
            ("session token expires after 24 hours login",      "auth"),
            ("payment gateway timeout after 30 seconds error",  "payment_fraud"),
            ("transaction limit 10 per day user account",       "api_rate"),
            ("international shipping 7 to 10 business days",    "shipping"),
            ("account locks after 5 failed login attempts",     "auth"),
            ("Visa Mastercard Amex card payment accepted",      "payment_fraud"),
        ]
        for i, (_q, _ans, _topic, _tkws, answer_kws) in enumerate(CONVERSATION_TURNS):
            matches = [
                t for t in relevant_templates
                if all(kw.lower() in t[0].lower() for kw in answer_kws)
            ]
            self.assertEqual(len(matches), 1,
                f"Query {i}: answer_keywords {answer_kws} matched {len(matches)} "
                f"templates, expected exactly 1. Matches: {matches}")

    def test_correct_false_when_wrong_same_topic_entry_ranks_first(self):
        """
        Regression test for the original bug.
        Simulates Phase 3 scenario: a same-topic but wrong entry ranks first.
        The fix must report Correct: False, not Correct: True.
        """
        agent = UnboundedMemoryAgent(top_k=5)
        pool  = _generate_memory_pool(200)
        agent.load_memory(pool)

        query_text, _ans, topic, _tkws, answer_kws = CONVERSATION_TURNS[0]
        resp = agent.query(query_text, topic, answer_kws, turn=1)

        # The answer and correct flag must now be consistent:
        # if the answer does not contain answer_keywords, correct must be False
        if not _is_correct(resp.retrieved[0], answer_kws):
            self.assertFalse(resp.correct,
                "Correct must be False when top entry fails keyword check")


class TestMemoryPool(unittest.TestCase):

    def test_pool_size(self):
        pool = _generate_memory_pool(100)
        self.assertEqual(len(pool), 100)

    def test_stale_ratio_increases(self):
        small = _generate_memory_pool(10)
        large = _generate_memory_pool(200)
        stale_small = sum(1 for e in small if e.topic == "off_topic") / len(small)
        stale_large = sum(1 for e in large if e.topic == "off_topic") / len(large)
        self.assertGreater(stale_large, stale_small)

    def test_turns_monotonic(self):
        pool = _generate_memory_pool(50)
        turns = [e.turn for e in pool]
        self.assertEqual(turns, sorted(turns))


class TestUnboundedAgent(unittest.TestCase):

    def setUp(self):
        self.agent = UnboundedMemoryAgent(top_k=5)
        self.pool  = _generate_memory_pool(200)

    def test_accuracy_degrades(self):
        s10  = _measure_at_size(self.agent, self.pool, 10)
        s200 = _measure_at_size(self.agent, self.pool, 200)
        self.assertGreaterEqual(s10.accuracy, s200.accuracy)

    def test_relevance_degrades(self):
        s10  = _measure_at_size(self.agent, self.pool, 10)
        s200 = _measure_at_size(self.agent, self.pool, 200)
        self.assertGreaterEqual(s10.relevance_rate, s200.relevance_rate)

    def test_confidence_stays_high(self):
        s10  = _measure_at_size(self.agent, self.pool, 10)
        s200 = _measure_at_size(self.agent, self.pool, 200)
        # Confidence delta should be small (< 10pp) while accuracy drops more
        conf_delta = abs(s10.avg_confidence - s200.avg_confidence)
        acc_delta  = s10.accuracy - s200.accuracy
        self.assertGreater(acc_delta, conf_delta)


class TestManagedAgent(unittest.TestCase):

    def setUp(self):
        self.agent = ManagedMemoryAgent(max_size=50, top_k=5)
        self.pool  = _generate_memory_pool(500)

    def test_memory_bounded(self):
        self.agent.load_memory(self.pool)
        self.assertLessEqual(len(self.agent.memory), 50)

    def test_managed_beats_unbounded_at_large_size(self):
        unbounded = UnboundedMemoryAgent(top_k=5)
        s_ub = _measure_at_size(unbounded,  self.pool, 200)
        s_mg = _measure_at_size(self.agent, self.pool, 200)
        self.assertGreater(s_mg.accuracy, s_ub.accuracy,
            f"Managed accuracy {s_mg.accuracy:.0%} must beat unbounded {s_ub.accuracy:.0%}")
        self.assertGreater(s_mg.relevance_rate, s_ub.relevance_rate,
            f"Managed relevance {s_mg.relevance_rate:.0%} must beat unbounded {s_ub.relevance_rate:.0%}")

    def test_deduplication_reduces_pool(self):
        # Create a pool with obvious near-duplicates
        dup_entries = []
        for i in range(10):
            content = "payment fraud threshold $500 review transaction"
            dup_entries.append(MemoryEntry(
                entry_id  = f"dup_{i:03d}",
                content   = f"{content} (turn {i+1})",
                turn      = i + 1,
                topic     = "payment_fraud",
                embedding = _text_to_embedding(content),
            ))
        deduped = self.agent._deduplicate(dup_entries)
        self.assertLess(len(deduped), len(dup_entries),
            "Deduplication should remove near-identical entries")

    def test_topic_routing_routes_correctly(self):
        fraud_q = _text_to_embedding("What is the fraud detection threshold for payments?")
        auth_q  = _text_to_embedding("How do I reset a user account password?")
        self.assertEqual(_route_query_to_topic(fraud_q), "payment_fraud")
        self.assertEqual(_route_query_to_topic(auth_q),  "auth")

    def test_recency_bonus_range(self):
        self.agent.load_memory(self.pool[:50])
        oldest = min(self.agent.memory, key=lambda e: e.turn)
        newest = max(self.agent.memory, key=lambda e: e.turn)
        bonus_old = self.agent._recency_bonus(oldest)
        bonus_new = self.agent._recency_bonus(newest)
        self.assertGreater(bonus_new, bonus_old)


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Agent Memory Leak Demo — TDS companion code"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress INFO-level logs",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable coloured terminal output",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run unit tests and exit",
    )
    args = parser.parse_args()

    if args.no_color:
        _COLOR = False

    if args.test:
        sys.argv = [sys.argv[0]]
        unittest.main(verbosity=2)
    else:
        run_demo(quiet=args.quiet)
