
#!/usr/bin/env python
"""A/B test of memory LAYOUT against real Anthropic Memory Stores.

Everything before this script measured our own reimplementation. This one
exercises the platform: two real stores hold byte-identical memory content and
differ only in how it is arranged and how the agent is told to read it.

  control : every memory flat at /memories/<id>.md, no read guidance
  regen   : memories at /s<k>/<topic>/<id>.md by retention stage, tails
            compressed to gist, and store `instructions` steering the agent to
            read /s0../s3/ first and descend only on failure

The agent reads the mount with ordinary file tools; we never inject evidence
into the prompt. So this tests the claim the paper actually makes about the
platform: does arranging memory by predicted relevance help a real agent?

Usage:
  python run_memory_stores_ab.py --build      # create stores (once)
  python run_memory_stores_ab.py --run -n 10  # ask questions, score
  python run_memory_stores_ab.py --teardown
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import anthropic

from regen_mem.graph import MemoryGraph, Params

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".ab_state.json"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
BETA = "agent-memory-2026-07-22"
SESSION_BETA = "managed-agents-2026-04-01"

REGEN_INSTRUCTIONS = (
    "Memories are filed by retention stage. /s0/ and /s1/ hold recent, "
    "full-detail memories; /s2/ and /s3/ hold older ones; /s4/ and deeper hold "
    "faded memories compressed to a short gist. ALWAYS list and read /s0/ "
    "through /s3/ first. Only descend into /s4/ and deeper if the shallow "
    "stages do not answer the question. Paths are grouped by topic, so list a "
    "stage directory before reading individual files."
)
CONTROL_INSTRUCTIONS = (
    "Memories are stored as files under /memories/. Read them to answer "
    "questions about the conversation history."
)

def client(beta: str = BETA) -> anthropic.Anthropic:
    return anthropic.Anthropic(default_headers={"anthropic-beta": beta})

def slug(s: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (s or "misc").lower()).strip("-")
    return (s[:n] or "misc")

# ---------------------------------------------------------------- build
def build(args) -> None:
    c = client()
    params = Params()
    g = MemoryGraph.load(
        ROOT / ".graphs" / "conv-26_None_claude-sonnet-4-5.json", params)
    last = max(x.t_days or 0 for x in g.contents.values())
    g.apply_decay(last)
    g.replay()

    nodes = list(g.contents.values())
    print(f"source graph: {len(nodes)} memories, "
          f"stages {min(n.stage for n in nodes)}..{max(n.stage for n in nodes)}")

    stores = {}
    for cond in ("control", "regen"):
        s = c.beta.memory_stores.create(
            name=f"REGEN-AB {cond}",
            description=("Conversation memory for Caroline and Melanie. "
                         + (REGEN_INSTRUCTIONS if cond == "regen"
                            else CONTROL_INSTRUCTIONS)))
        stores[cond] = s.id
        print(f"  {cond:8s} -> {s.id}")

    t0 = time.time()
    for i, n in enumerate(nodes):
        # control: flat, always full text
        c.beta.memory_stores.memories.create(
            stores["control"],
            path=f"/memories/{slug(n.cid)}.md",
            content=n.text)
        # regen: staged path, tail compressed exactly as the lifecycle would
        c.beta.memory_stores.memories.create(
            stores["regen"],
            path=f"/s{n.stage}/{slug(n.topic or n.tag)}/{slug(n.cid)}.md",
            content=n.visible_text())
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(nodes)} written ({time.time()-t0:.0f}s)")

    # byte accounting: the regen store is smaller because its tail is gist
    sizes = {k: sum(len(n.text) if k == "control" else len(n.visible_text())
                    for n in nodes) for k in stores}
    STATE.write_text(json.dumps({"stores": stores, "sizes": sizes}, indent=1))
    print(f"\ncontent bytes  control={sizes['control']:,}  "
          f"regen={sizes['regen']:,}  ({100*sizes['regen']/sizes['control']:.0f}%)")
    print(f"state -> {STATE}")

# ---------------------------------------------------------------- run
JUDGE = """Grade this answer. Output valid JSON only.

Accept paraphrase, different surface form, equivalent dates. Reject wrong
entities, wrong dates, wrong counts, and refusals.

QUESTION: {q}
GOLD: {gold}
PREDICTION: {pred}

{{"correct": true|false}}"""

def judge(jc, q, gold, pred) -> bool:
    if not (pred or "").strip():
        return False
    try:
        r = jc.messages.create(
            model="claude-haiku-4-5", max_tokens=200,
            messages=[{"role": "user",
                       "content": JUDGE.format(q=q, gold=gold, pred=pred)},
                      {"role": "assistant", "content": "{"}])
        txt = "{" + "".join(b.text for b in r.content if b.type == "text")
        return bool(json.loads(txt, strict=False).get("correct"))
    except Exception as e:
        print(f"\n  ! JUDGE FAILED: {type(e).__name__}: {str(e)[:120]}")
        raise

def ensure_agent(c):
    st = json.loads(STATE.read_text())
    if st.get("agent") and st.get("env"):
        return st["agent"], st["env"]
    env = c.beta.environments.create(name="regen-ab-env")
    agent = c.beta.agents.create(
        name="regen-ab-agent", model="claude-sonnet-4-5",
        tools=[{"type": "agent_toolset_20260401"}],
        system=("Answer questions about a past conversation using ONLY the "
                "memory files mounted in your sandbox. Explore the mount, read "
                "what you need, then answer. Give the minimal concrete entity, "
                "phrase, number or date. Do not write a sentence. For dates, "
                "give the specific date as stated in the memory."))
    st["agent"], st["env"] = agent.id, env.id
    STATE.write_text(json.dumps(st, indent=1))
    print(f"agent={agent.id} env={env.id}")
    return agent.id, env.id

def ask(c, agent_id, env_id, store_id, instructions, question, timeout=240):
    """One question in a fresh session so nothing leaks between questions.

    The agent reads the mount with its own file tools; we never inject
    evidence. We poll the event stream until the session goes idle and take
    the last agent.message as the answer.
    """
    sess = c.beta.sessions.create(
        agent=agent_id, environment_id=env_id,
        resources=[{"type": "memory_store", "memory_store_id": store_id,
                    "access": "read_only", "instructions": instructions}],
        initial_events=[{"type": "user.message",
                         "content": [{"type": "text", "text": question}]}])
    t0 = time.time()
    data = []
    while time.time() - t0 < timeout:
        data = list(c.beta.sessions.events.list(sess.id).data)
        if any(getattr(e, "type", "") == "session.status_idle" for e in data):
            break
        time.sleep(3)

    text, tok, tools = "", 0, 0
    for e in data:
        et = getattr(e, "type", "")
        if et == "agent.tool_use":
            tools += 1
        elif et == "agent.message":
            body = getattr(e, "content", None)
            if isinstance(body, list):
                text = " ".join(getattr(b, "text", "") for b in body).strip()
        elif et == "session.usage":
            u = getattr(e, "usage", None) or e
            tok = (getattr(u, "input_tokens", 0) or 0) + (getattr(u, "output_tokens", 0) or 0)
    return text, tok, tools

def run(args) -> None:
    st = json.loads(STATE.read_text())
    c = client(SESSION_BETA)
    jc = anthropic.Anthropic()   # no beta header: messages.create rejects it
    agent_id, env_id = ensure_agent(c)

    sample = next(s for s in json.loads((ROOT / "data" / "locomo10.json").read_text())
                  if s["sample_id"] == "conv-26")
    g = MemoryGraph.load(ROOT / ".graphs" / "conv-26_None_claude-sonnet-4-5.json", Params())
    kept = {d for x in g.contents.values() for d in x.dia_ids}
    qa = [q for q in sample["qa"] if q.get("category") != 5 and q.get("evidence")
          and all(e in kept for e in q["evidence"])]
    random.Random(0).shuffle(qa)
    qa = qa[: args.n]
    print(f"{len(qa)} questions, 2 conditions, fresh session each\n")

    rows = []
    for cond, instr in (("control", CONTROL_INSTRUCTIONS),
                        ("regen", REGEN_INSTRUCTIONS)):
        ok = toks = 0
        for i, q in enumerate(qa):
            try:
                pred, tok, ntools = ask(c, agent_id, env_id, st["stores"][cond],
                                        instr, q["question"])
            except Exception as e:
                print(f"  ! {cond} q{i}: {type(e).__name__}: {str(e)[:120]}")
                pred, tok, ntools = "", 0, 0
            good = judge(jc, q["question"], str(q["answer"]), pred)
            ok += good; toks += tok
            rows.append({"condition": cond, "q": q["question"],
                         "gold": str(q["answer"]), "pred": pred,
                         "correct": good, "tokens": tok, "tool_calls": ntools,
                         "category": q.get("category")})
            print(f"  {cond:8s} {i+1}/{len(qa)} acc={ok/(i+1):.0%}", end="\r")
        print(f"  {cond:8s} acc={100*ok/len(qa):5.1f}%  tok/q={toks//max(1,len(qa)):,}      ")

    (RESULTS / "memory_stores_ab.json").write_text(json.dumps(rows, indent=1))
    print("\n" + "=" * 54)
    for cond in ("control", "regen"):
        rs = [r for r in rows if r["condition"] == cond]
        acc = 100 * sum(r["correct"] for r in rs) / len(rs)
        tk = sum(r["tokens"] for r in rs) / len(rs)
        tc = sum(r.get("tool_calls", 0) for r in rs) / len(rs)
        print(f"  {cond:8s}  acc={acc:5.1f}%   tok/q={tk:8,.0f}   tool_calls/q={tc:4.1f}")
    print("=" * 54)
    print(f"\nwrote results/memory_stores_ab.json")

def teardown(args) -> None:
    st = json.loads(STATE.read_text())
    c = client()
    for cond, sid in st.get("stores", {}).items():
        try:
            c.beta.memory_stores.delete(sid); print(f"  deleted {cond} {sid}")
        except Exception as e:
            print(f"  {cond}: {type(e).__name__}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--teardown", action="store_true")
    ap.add_argument("-n", type=int, default=10)
    a = ap.parse_args()
    if a.build: build(a)
    if a.run: run(a)
    if a.teardown: teardown(a)
