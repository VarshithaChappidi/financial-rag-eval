import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, TypedDict

import anthropic
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from agent.tools import calculate, search_filings
from rag.generate import SYSTEM as BASE_SYSTEM

load_dotenv()
MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
TRACE = Path("data/agent_trace.jsonl")
MAX_ATTEMPTS = 2

SYSTEM = (
    BASE_SYSTEM
    + " For any arithmetic (differences, growth rates, sums) call the calculate tool; "
    "never compute numbers yourself. Quote source numbers exactly as written."
)
TOOLS = [{
    "name": "calculate",
    "description": "Evaluate an arithmetic expression using numbers, + - * / ** and parentheses.",
    "input_schema": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
}]
ADVICE = re.compile(
    r"\b(should i (buy|sell|invest)|which stock|price target|buy or sell|good investment)\b", re.I
)


class State(TypedDict, total=False):
    question: str
    sources: list
    calcs: list
    answer: str
    unsupported: list
    attempts: int
    status: str  # ok | unverified | refused | rejected


def trace(event: str, **data) -> None:
    TRACE.parent.mkdir(exist_ok=True)
    with TRACE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"t": round(time.time(), 1), "event": event, **data}, default=str) + "\n")


def guard(state: State) -> dict:
    refused = bool(ADVICE.search(state["question"]))
    trace("guard", refused=refused)
    if refused:
        return {
            "status": "refused",
            "answer": "I can answer questions about what the filings say, but I can't give investment advice.",
        }
    return {}


def retrieve(state: State) -> dict:
    sources = search_filings(state["question"])
    trace("retrieve", ids=[c["id"] for c in sources])
    return {"sources": sources, "calcs": [], "attempts": 0, "unsupported": []}


def draft(state: State) -> dict:
    context = "\n\n".join(
        f"[{i + 1}] ({c['ticker']} | {c['section']})\n{c['text']}" for i, c in enumerate(state["sources"])
    )
    user = f"Sources:\n{context}\n\nQuestion: {state['question']}"
    if state.get("unsupported"):
        user += (
            "\n\nYour previous draft contained numbers that are not in the sources or calculator "
            f"results: {state['unsupported']}. Rewrite using only supported numbers."
        )
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user}]
    calcs = []
    msg = None
    for _ in range(4):
        msg = client.messages.create(
            model=MODEL, max_tokens=4000, system=SYSTEM, tools=TOOLS, messages=messages
        )
        if msg.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant", "content": msg.content})
        results = []
        for b in msg.content:
            if b.type == "tool_use":
                try:
                    val = calculate(b.input["expression"])
                    out = str(val)
                    calcs.append({"expression": b.input["expression"], "result": val})
                except Exception as e:
                    out = f"error: {e}"
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
        messages.append({"role": "user", "content": results})
    text = "".join(b.text for b in msg.content if b.type == "text") or "(no text returned)"
    attempts = state.get("attempts", 0) + 1
    trace("draft", attempt=attempts, calcs=calcs)
    return {"answer": text, "calcs": state.get("calcs", []) + calcs, "attempts": attempts}


NUM = re.compile(r"\d[\d,]*\.?\d*")


def verify(state: State) -> dict:
    text = re.sub(r"\[\d+\]", " ", state["answer"])           # citation markers
    text = re.sub(r"(?m)^\s*\d+[.)]\s", "", text)             # numbered-list markers
    text = re.sub(r"\b(19|20)\d{2}\b", " ", text)             # years
    source_text = " ".join(c["text"] for c in state["sources"]).replace(",", "")
    results = [float(c["result"]) for c in state["calcs"]]
    bad = []
    for m in NUM.findall(text):
        s = m.replace(",", "").rstrip(".")
        if not s:
            continue
        if re.search(rf"(?<![\d.]){re.escape(s)}(?!\d|\.\d)", source_text):
            continue
        try:
            v = float(s)
        except ValueError:
            continue
        if any(abs(v - r) <= max(0.051, abs(r) * 0.005) for r in results):
            continue
        bad.append(m)
    trace("verify", unsupported=bad)
    return {"unsupported": bad}


def route_after_verify(state: State) -> str:
    if state["unsupported"] and state["attempts"] < MAX_ATTEMPTS:
        return "draft"
    return "approve"


def cli_approver(state: State) -> bool:
    print("\n--- DRAFT ANSWER ---\n" + state["answer"])
    if state["unsupported"]:
        print("\nWARNING: unverified numbers:", state["unsupported"])
    return input("\nApprove this answer? [y/N] ").strip().lower() == "y"


def make_approve(approver: Callable[[State], bool]):
    def approve(state: State) -> dict:
        status = "unverified" if state["unsupported"] else "ok"
        ok = approver(state)
        trace("approve", approved=ok, status=status)
        return {"status": status if ok else "rejected"}
    return approve


def build(approver: Callable[[State], bool] = cli_approver):
    g = StateGraph(State)
    g.add_node("guard", guard)
    g.add_node("retrieve", retrieve)
    g.add_node("draft", draft)
    g.add_node("verify", verify)
    g.add_node("approve", make_approve(approver))
    g.add_edge(START, "guard")
    g.add_conditional_edges(
        "guard", lambda s: END if s.get("status") == "refused" else "retrieve",
        {"retrieve": "retrieve", END: END},
    )
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"draft": "draft", "approve": "approve"})
    g.add_edge("approve", END)
    return g.compile()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--auto", action="store_true", help="approve automatically")
    a = p.parse_args()
    app = build((lambda s: True) if a.auto else cli_approver)
    final = app.invoke({"question": a.question})
    print(f"\nSTATUS: {final['status']}\n\n{final['answer']}")
    for c in final.get("calcs", []):
        print(f"  calc: {c['expression']} = {c['result']}")