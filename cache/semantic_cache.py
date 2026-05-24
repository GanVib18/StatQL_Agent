import json
import time
import numpy as np
import faiss
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from sentence_transformers import SentenceTransformer

from agent.agent import AnalyticsAgent


class SemanticCache:
    """all-MiniLM-L6-v2 + FAISS. Persists across restarts. Tracks hit/miss analytics."""

    def __init__(self, threshold: float = 0.92,
                 cache_path: str = "cache_meta.json",
                 index_path: str = "cache_index.faiss"):
        self.threshold  = threshold
        self.cache_path = cache_path
        self.index_path = index_path
        self.model      = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim        = 384
        self.index      = faiss.IndexFlatIP(self.dim)
        self.metadata:  List[dict] = []
        self.stats      = {"hits": 0, "misses": 0, "lat_cached": [], "lat_fresh": []}
        self._load()

    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True).astype("float32")

    def _load(self):
        if Path(self.cache_path).exists():
            with open(self.cache_path) as f:
                self.metadata = json.load(f)
        if Path(self.index_path).exists() and self.metadata:
            self.index = faiss.read_index(self.index_path)
            print(f"✓ SemanticCache: loaded {len(self.metadata)} entries from disk")

    def _save(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.metadata, f, default=str, indent=2)
        if self.index.ntotal > 0:
            faiss.write_index(self.index, self.index_path)

    def get(self, question: str) -> Optional[dict]:
        if self.index.ntotal == 0:
            return None
        scores, idxs = self.index.search(self._embed(question), k=1)
        score, idx   = float(scores[0][0]), int(idxs[0][0])
        if score >= self.threshold and idx < len(self.metadata):
            self.stats["hits"] += 1
            return {**self.metadata[idx], "cache_hit": True, "similarity": round(score, 4)}
        return None

    def set(self, question: str, result: dict):
        self.index.add(self._embed(question))
        self.metadata.append({
            "question":  question,
            "answer":    result.get("answer"),
            "sql":       result.get("sql"),
            "stats":     result.get("stats"),
            "timestamp": datetime.now().isoformat(),
        })
        self._save()
        self.stats["misses"] += 1

    def invalidate_all(self):
        self.index    = faiss.IndexFlatIP(self.dim)
        self.metadata = []
        self._save()
        print("✓ Cache cleared.")

    def get_analytics(self) -> dict:
        total   = self.stats["hits"] + self.stats["misses"]
        lc, lf  = self.stats["lat_cached"], self.stats["lat_fresh"]
        avg_c   = float(np.mean(lc)) if lc else 0.0
        avg_f   = float(np.mean(lf)) if lf else 0.0
        return {
            "total_queries":         total,
            "cache_hits":            self.stats["hits"],
            "cache_misses":          self.stats["misses"],
            "hit_rate":              round(self.stats["hits"] / total, 3) if total else 0,
            "avg_latency_cached_s":  round(avg_c, 3),
            "avg_latency_fresh_s":   round(avg_f, 3),
            "latency_reduction_pct": round((1 - avg_c / avg_f) * 100, 1) if avg_f > 0 else 0,
            "llm_calls_saved":       self.stats["hits"],
        }


class CachedAgent:
    """Wraps AnalyticsAgent with a SemanticCache look-aside layer."""

    def __init__(self, agent: AnalyticsAgent, cache: SemanticCache):
        self.agent = agent
        self.cache = cache

    def ask(self, question: str) -> dict:
        t0     = time.perf_counter()
        cached = self.cache.get(question)
        if cached:
            lat = time.perf_counter() - t0
            self.cache.stats["lat_cached"].append(lat)
            return {**cached, "latency_s": round(lat, 4)}
        result = self.agent.ask(question)
        lat    = time.perf_counter() - t0
        self.cache.stats["lat_fresh"].append(lat)
        self.cache.set(question, result)
        return {**result, "cache_hit": False, "latency_s": round(lat, 4)}
