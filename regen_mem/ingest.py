"""Memory construction: distil dialogue into a Cue-Tag-Content graph.

One LLM call per session produces episodes with cues, tags, topics and a
context descriptor.  The context descriptor plus the session timestamp become
the contextual signal vector s* -- the paper's point that s* is nearly free,
since distillation pipelines already extract these fields and then discard
them after cue construction.

The same graph is shared by REGEN and the baselines, so retrieval differences
are not confounded by construction differences.
"""
from __future__ import annotations

import json
import re
from datetime import datetime

import numpy as np

from .graph import Content, MemoryGraph, Params
from .llm import LLM
from .signals import SignalVector, unit

CTX_DIMS = 256  # truncated embedding dimensionality for the context term

SESSION_PROMPT = """You are a memory distillation system. Output valid JSON only.

Given one session of a conversation, produce episodic memory units.

For EACH episode, output:
  "dia_ids": the dialogue ids covered (e.g. ["D1:3","D1:4"]) - use ONLY ids present in the input
  "text":    a self-contained sentence stating what happened, with pronouns resolved
             to names AND every relative time reference resolved to an absolute
             date using SESSION DATE below. Write "on 7 May 2023", never
             "yesterday"/"last week"/"on Friday". If the episode states when
             something happened, that date MUST appear in this sentence -- a
             reader with no other context must be able to answer "when did this
             happen?" from the text alone.
  "gist":    a terse 3-10 word summary that KEEPS any date or number from the
             text (this is all that survives compression, so detail dropped
             here is gone for good)
  "tag":     a 1-3 word associative relation label (e.g. "support group attendance")
  "cues":    2-8 fine-grained keywords appearing in or directly naming the content
             (entities, activities, objects, places, people)
  "topic":   a short recurring-theme label shared across related episodes
  "context": a descriptor of the CIRCUMSTANCES: participants, location,
             activity, mood, setting. Not the content itself.

Merge a question and its answer into one episode. Prefer several small
episodes over one large one. Cover the whole session.

Also output "semantic": stable facts about a person (preferences, attributes,
relationships) as objects with "person", "aspect", "fact", "cues". Resolve any
time reference in "fact" to an absolute date as well.

JSON schema:
{"episodes":[{"dia_ids":[],"text":"","gist":"","tag":"","cues":[],"topic":"","context":""}],
 "semantic":[{"person":"","aspect":"","fact":"","cues":[]}]}

SESSION DATE: {date}
SPEAKERS: {speakers}

DIALOGUE:
{dialogue}
"""


def parse_locomo_date(s: str) -> datetime | None:
    """LoCoMo timestamps look like '1:56 pm on 8 May, 2023'."""
    if not s:
        return None
    m = re.search(r"on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", s)
    if not m:
        return None
    day, mon, year = m.group(1), m.group(2)[:3], m.group(3)
    for fmt in ("%d %b %Y",):
        try:
            return datetime.strptime(f"{day} {mon} {year}", fmt)
        except ValueError:
            pass
    return None


def build_graph(
    sample: dict, llm: LLM, params: Params, *, max_sessions: int | None = None
) -> MemoryGraph:
    conv = sample["conversation"]
    speakers = f"{conv.get('speaker_a','A')} and {conv.get('speaker_b','B')}"

    sess_keys = sorted(
        (k for k in conv if re.fullmatch(r"session_\d+", k)),
        key=lambda k: int(k.split("_")[1]),
    )
    if max_sessions:
        sess_keys = sess_keys[:max_sessions]

    dates: dict[str, datetime | None] = {
        k: parse_locomo_date(conv.get(f"{k}_date_time", "")) for k in sess_keys
    }
    valid = [d for d in dates.values() if d]
    epoch = min(valid) if valid else datetime(2023, 1, 1)

    g = MemoryGraph(params)
    pending: list[tuple[Content, str]] = []  # (content, context descriptor)

    for sk in sess_keys:
        turns = conv.get(sk) or []
        if not turns:
            continue
        dt = dates.get(sk)
        t_days = (dt - epoch).days if dt else None
        dialogue = "\n".join(
            f"[{t.get('dia_id','?')}] {t.get('speaker','?')}: {t.get('text','')}"
            for t in turns
        )
        prompt = (
            SESSION_PROMPT
            .replace("{date}", conv.get(f"{sk}_date_time", "unknown"))
            .replace("{speakers}", speakers)
            .replace("{dialogue}", dialogue)
        )
        try:
            data = llm.complete_json(prompt, phase="construction", max_tokens=16000)
        except Exception as e:  # noqa: BLE001
            print(f"    ! session {sk} distillation failed: {type(e).__name__}")
            continue

        for i, ep in enumerate(data.get("episodes", []) or []):
            cid = f"{sk}_e{i}"
            ctx = (ep.get("context") or "").strip()
            c = Content(
                cid=cid, kind="episode",
                text=(ep.get("text") or "").strip(),
                gist=(ep.get("gist") or "").strip(),
                tag=(ep.get("tag") or "").strip(),
                cues=[str(x) for x in (ep.get("cues") or [])],
                topic=(ep.get("topic") or "").strip(),
                signal=SignalVector(t_days=t_days, ctx_emb=np.zeros(1, dtype=np.float32)),
                t_days=t_days,
                dia_ids=[str(x) for x in (ep.get("dia_ids") or [])],
            )
            if c.text:
                pending.append((c, f"{ctx} | {c.topic} | {speakers}"))

        # Semantic units are distilled from the same session as the episodes,
        # so they inherit that session's dialogue ids.  Without this they carry
        # no dia_ids at all and a correct answer sourced from semantic memory
        # scores ev_recall = 0, making ~47% of the store unmeasurable.
        session_dia = [str(t.get("dia_id")) for t in turns if t.get("dia_id")]
        for j, sm in enumerate(data.get("semantic", []) or []):
            cid = f"{sk}_s{j}"
            person = (sm.get("person") or "").strip()
            aspect = (sm.get("aspect") or "").strip()
            fact = (sm.get("fact") or "").strip()
            if not fact:
                continue
            c = Content(
                cid=cid, kind="semantic",
                text=f"{person}: {fact}" if person else fact,
                gist=f"{person} {aspect}".strip(),
                tag=aspect,
                cues=[str(x) for x in (sm.get("cues") or [])] + ([person] if person else []),
                topic=aspect,
                signal=SignalVector(t_days=t_days, ctx_emb=np.zeros(1, dtype=np.float32)),
                t_days=t_days,
                dia_ids=list(sm.get("dia_ids") or session_dia),
            )
            pending.append((c, f"{person} {aspect} | {speakers}"))

    # Embed all context descriptors in one batched pass.
    if pending:
        embs = llm.embed([d for _, d in pending], phase="construction_embed")
        for (c, desc), e in zip(pending, embs):
            c.signal = SignalVector(
                t_days=c.t_days,
                ctx_emb=unit(np.asarray(e[:CTX_DIMS], dtype=np.float32)),
                descriptor=desc,
            )
            g.add(c)

    return g


QUERY_SIGNAL_PROMPT = """Output valid JSON only.

From the question, extract the retrieval signal:
  "cues":    2-8 keywords likely to appear in the stored memory
  "context": a descriptor of the circumstances the question is about
             (participants, activity, setting) - not the question wording
  "date":    an explicit date the question anchors to as YYYY-MM-DD, else null
  "temporal": true if the question is about WHEN something happened or about
              ordering/duration, else false

{"cues":[],"context":"","date":null,"temporal":false}

QUESTION: {q}
"""


def query_signal(
    question: str, llm: LLM, epoch: datetime, *, fallback_now: float | None
) -> tuple[SignalVector, list[str], bool]:
    """Build the query-side signal s_r."""
    try:
        d = llm.complete_json(
            QUERY_SIGNAL_PROMPT.replace("{q}", question),
            phase="retrieval", max_tokens=2000,
        )
    except Exception:  # noqa: BLE001
        d = {}

    cues = [str(x) for x in (d.get("cues") or [])] or question.split()[:6]
    ctx = (d.get("context") or question).strip()
    temporal = bool(d.get("temporal"))

    t_days: float | None = None
    if d.get("date"):
        try:
            t_days = (datetime.strptime(str(d["date"]), "%Y-%m-%d") - epoch).days
        except ValueError:
            t_days = None

    emb = llm.embed([ctx], phase="retrieval_embed")[0]
    return (
        SignalVector(
            t_days=t_days,
            ctx_emb=unit(np.asarray(emb[:CTX_DIMS], dtype=np.float32)),
            descriptor=ctx,
        ),
        cues,
        temporal,
    )
