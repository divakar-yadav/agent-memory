"""LLM access layer: disk-cached calls + token accounting.

Every call is cached on (model, prompt, schema) so re-runs of an experiment
cost nothing.  This matters a lot: memory construction dominates cost, and we
re-run retrieval many times against the same graph.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    import anthropic
except ImportError:  # optional provider
    anthropic = None

CACHE_DIR = Path(__file__).resolve().parent.parent / ".llm_cache"
CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class Usage:
    """Token accounting, split by phase so we can report construction vs retrieval."""

    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_phase: dict[str, dict[str, int]] = field(default_factory=dict)
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    def add(self, phase: str, tin: int, tout: int, cached: bool) -> None:
        with self._lock:
            self.calls += 1
            if cached:
                self.cached += 1
            self.input_tokens += tin
            self.output_tokens += tout
            p = self.by_phase.setdefault(
                phase, {"calls": 0, "input_tokens": 0, "output_tokens": 0}
            )
            p["calls"] += 1
            p["input_tokens"] += tin
            p["output_tokens"] += tout

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "cache_hits": self.cached,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
                "by_phase": json.loads(json.dumps(self.by_phase)),
            }

    def reset(self) -> None:
        with self._lock:
            self.calls = self.cached = self.input_tokens = self.output_tokens = 0
            self.by_phase = {}


USAGE = Usage()


def _key(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:40]


def _provider_for(model: str) -> str:
    return "anthropic" if model.startswith("claude") else "openai"


class LLM:
    """One interface over both vendors.

    Completions dispatch on the model name.  Embeddings always use OpenAI:
    Anthropic ships no embedding endpoint, and holding the embedding model
    fixed across backbones is what keeps the retrieval substrate constant so
    that a backbone swap changes only the reasoning.
    """

    def __init__(self, model: str, embed_model: str = "text-embedding-3-small"):
        self.model = model
        self.embed_model = embed_model
        self.provider = _provider_for(model)
        self.client = OpenAI()          # always needed, for embeddings
        self.anthropic = None
        if self.provider == "anthropic":
            if anthropic is None:
                raise RuntimeError("pip install anthropic")
            self.anthropic = anthropic.Anthropic()

    # ---------------------------------------------------------------- text
    def complete(
        self,
        prompt: str,
        *,
        phase: str = "misc",
        system: str | None = None,
        max_tokens: int = 4096,
        as_json: bool = False,
        retries: int = 4,
    ) -> str:
        ck = _key(self.model, system, prompt, max_tokens, as_json)
        cf = CACHE_DIR / f"c_{ck}.json"
        if cf.exists():
            d = json.loads(cf.read_text())
            USAGE.add(phase, d["tin"], d["tout"], cached=True)
            return d["text"]

        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        last: Exception | None = None
        for attempt in range(retries):
            try:
                if self.provider == "anthropic":
                    akw: dict[str, Any] = {
                        "model": self.model,
                        "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    if system:
                        akw["system"] = system
                    if as_json:
                        # No json_object mode on this API; prefill an opening
                        # brace so the model continues a JSON object instead of
                        # narrating one.  _parse_json re-attaches it.
                        akw["messages"].append(
                            {"role": "assistant", "content": "{"}
                        )
                    ar = self.anthropic.messages.create(**akw)
                    text = "".join(
                        b.text for b in ar.content if getattr(b, "type", "") == "text"
                    )
                    if as_json and not text.lstrip().startswith("{"):
                        text = "{" + text
                    tin = ar.usage.input_tokens
                    tout = ar.usage.output_tokens
                    cf.write_text(json.dumps({"text": text, "tin": tin, "tout": tout}))
                    USAGE.add(phase, tin, tout, cached=False)
                    return text

                kw: dict[str, Any] = {
                    "model": self.model,
                    "input": msgs,
                    "max_output_tokens": max_tokens,
                }
                if as_json:
                    kw["text"] = {"format": {"type": "json_object"}}
                r = self.client.responses.create(**kw)
                text = r.output_text or ""
                tin = r.usage.input_tokens if r.usage else 0
                tout = r.usage.output_tokens if r.usage else 0
                cf.write_text(json.dumps({"text": text, "tin": tin, "tout": tout}))
                USAGE.add(phase, tin, tout, cached=False)
                return text
            except Exception as e:  # noqa: BLE001 - retry on any transport/API error
                last = e
                time.sleep(1.5 * (2**attempt))
        raise RuntimeError(f"LLM failed after {retries} tries: {last}")

    def complete_json(self, prompt: str, **kw: Any) -> Any:
        """Complete and parse JSON, tolerating fenced output."""
        raw = self.complete(prompt, as_json=True, **kw)
        return _parse_json(raw)

    # ----------------------------------------------------------- embedding
    def embed(self, texts: list[str], *, phase: str = "embed") -> list[list[float]]:
        out: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
        todo: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            cf = CACHE_DIR / f"e_{_key(self.embed_model, t)}.json"
            if cf.exists():
                out[i] = json.loads(cf.read_text())
            else:
                todo.append((i, t))

        for s in range(0, len(todo), 128):
            batch = todo[s : s + 128]
            r = self.client.embeddings.create(
                model=self.embed_model, input=[t for _, t in batch]
            )
            USAGE.add(phase, r.usage.total_tokens, 0, cached=False)
            for (i, t), item in zip(batch, r.data):
                out[i] = item.embedding
                (CACHE_DIR / f"e_{_key(self.embed_model, t)}.json").write_text(
                    json.dumps(item.embedding)
                )
        return out  # type: ignore[return-value]


def _parse_json(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    # strict=False tolerates literal newlines inside string values, which
    # models emit when they write prose into a JSON field.
    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError:
        pass
    # Salvage the outermost object. Arrays are tried last and only when no
    # object parses, so a stray "actions": [...] can never be mistaken for
    # the whole response.
    for a, b in (("{", "}"), ("[", "]")):
        i, j = raw.find(a), raw.rfind(b)
        if i != -1 and j > i:
            try:
                return json.loads(raw[i : j + 1], strict=False)
            except json.JSONDecodeError:
                continue
    # Last resort: pull the fields we actually need out of the text.
    out: dict = {}
    m = re.search(r'"mode"\s*:\s*"(\w+)"', raw)
    if m:
        out["mode"] = m.group(1)
    m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if m:
        out["answer"] = m.group(1)
    cues = re.search(r'"new_cues"\s*:\s*\[([^\]]*)\]', raw)
    if cues:
        out["new_cues"] = re.findall(r'"([^"]+)"', cues.group(1))
    acts = re.search(r'"actions"\s*:\s*\[([^\]]*)\]', raw)
    if acts:
        out["actions"] = re.findall(r'"([^"]+)"', acts.group(1))
    if out:
        return out
    raise ValueError("unparseable model output")
