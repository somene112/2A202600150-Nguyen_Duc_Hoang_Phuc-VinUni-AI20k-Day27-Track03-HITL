"""Exercise 4 - Structured SQLite audit trail + durable checkpointer."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import db_conn, db_path, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)


console = Console()
AGENT_ID = "pr-review-agent@v0.1"


async def audit(state, entry: AuditEntry) -> None:
    """Write one structured AuditEntry row to the audit_events table."""
    await write_audit_event(thread_id=state["thread_id"], pr_url=state["pr_url"], entry=entry)


async def audit_once_before_interrupt(state, entry: AuditEntry) -> None:
    """Avoid duplicate pre-interrupt rows when LangGraph re-enters a node on resume."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT 1
              FROM audit_events
             WHERE thread_id = ?
               AND action = ?
               AND decision = ?
               AND reviewer_id IS NULL
             LIMIT 1
            """,
            (state["thread_id"], entry.action, entry.decision),
        ) as cur:
            exists = await cur.fetchone()
    if exists is None:
        await audit(state, entry)


def _elapsed_ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def _files_text(state) -> str:
    return "\n".join(f"- {path}" for path in state.get("pr_files", [])) or "(not available)"


async def node_fetch_pr(state):
    console.print("[cyan]-> fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]OK[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="fetch_pr",
        confidence=0.0,
        risk_level="med",
        decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }


async def node_analyze(state):
    console.print("[cyan]-> analyze[/cyan]")
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        a: PRAnalysis = await llm.ainvoke([
            {
                "role": "system",
                "content": (
                    "You are a senior code reviewer. Return structured PRAnalysis. "
                    f"If confidence is below {ESCALATE_THRESHOLD:.2f}, generate 2-4 "
                    "specific escalation_questions that reference concrete files, diff "
                    "sections, or security risks from the PR. Keep ordinary schema or product "
                    "changes with unresolved migration/testing questions in the human-review "
                    "band. Do not set confidence below "
                    f"{ESCALATE_THRESHOLD:.2f} solely because a schema/product addition is "
                    "missing a migration or tests; use about 0.60-0.72 for that case. If the diff "
                    "includes a cluster of critical security risks such as MD5 password hashing, "
                    "plaintext token storage, SQL injection, or hard-coded user_id values, treat "
                    f"the review as high risk and set confidence below {ESCALATE_THRESHOLD:.2f} "
                    "unless the diff itself clearly mitigates those risks."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Title: {state['pr_title']}\n\n"
                    f"Changed files:\n{_files_text(state)}\n\n"
                    f"Diff:\n{state['pr_diff']}"
                ),
            },
        ])
    console.print(f"  [green]OK[/green] confidence={a.confidence:.0%}, {len(a.comments)} comment(s)")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="analyze",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=a.confidence_reasoning,
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"analysis": a}


async def node_route(state):
    console.print("[cyan]-> route[/cyan]")
    t0 = time.monotonic()
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]OK[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="route",
        confidence=c,
        risk_level=risk_level_for(c),
        decision=decision,
        reason=f"Routed by confidence thresholds: auto>={AUTO_APPROVE_THRESHOLD:.2f}, escalate<{ESCALATE_THRESHOLD:.2f}",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"decision": decision}


async def node_human_approval(state):
    console.print("[cyan]-> human_approval[/cyan]")
    t0 = time.monotonic()
    a = state["analysis"]

    await audit_once_before_interrupt(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=None,
        decision="pending",
        reason="Waiting for reviewer approval",
        execution_time_ms=_elapsed_ms(t0),
    ))

    resp = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })

    choice = resp.get("choice") or "reject"
    feedback = resp.get("feedback") or f"Reviewer selected {choice}"
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision=choice,
        reason=feedback,
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"human_choice": choice, "human_feedback": resp.get("feedback")}


def _render_comment_body(state) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` - {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]POSTED[/green] GitHub PR comment to {state['pr_url']}")
        return "committed"
    except Exception as e:
        console.print(f"  [red]FAIL[/red] post failed: {e}")
        return "commit_failed"


async def node_commit(state):
    console.print("[cyan]-> commit[/cyan]")
    t0 = time.monotonic()
    a = state["analysis"]
    if state.get("escalation_answers") or state.get("human_choice") == "approve":
        action = _post(state)
    else:
        console.print(f"  [yellow].[/yellow] skipping comment (choice={state.get('human_choice')})")
        action = "rejected"

    if state.get("escalation_answers"):
        decision = "escalate"
    else:
        decision = state.get("human_choice") or "pending"
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="commit",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision=decision,
        reason=f"Final action: {action}",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"final_action": action}


async def node_auto_approve(state):
    console.print("[cyan]-> auto_approve[/cyan]  [dim]high confidence - posting directly[/dim]")
    t0 = time.monotonic()
    a = state["analysis"]
    action = _post(state)
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="auto_approve",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="auto",
        reason=f"Posted without human review. Final action: {action}",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"final_action": f"auto_{action}"}


async def node_escalate(state):
    console.print("[cyan]-> escalate[/cyan]")
    t0 = time.monotonic()
    a = state["analysis"]
    questions = a.escalation_questions or [
        "What is the intent of this PR?",
        "Are there security, migration, or deployment concerns not visible in the diff?",
    ]

    await audit_once_before_interrupt(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=None,
        decision="escalate",
        reason="Questions asked: " + " | ".join(questions),
        execution_time_ms=_elapsed_ms(t0),
    ))

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision="escalate",
        reason=f"Reviewer answers received for {len(answers)} escalation question(s)",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"escalation_answers": answers}


async def node_synthesize(state):
    console.print("[cyan]-> synthesize[/cyan]")
    t0 = time.monotonic()
    initial = state["analysis"]
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in (state.get("escalation_answers") or {}).items())
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke([
            {
                "role": "system",
                "content": (
                    "You are a senior code reviewer. Refine the original PR review using "
                    "the human reviewer's answers. Return a complete structured PRAnalysis."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original diff:\n{state['pr_diff']}\n\n"
                    f"Initial summary: {initial.summary}\n"
                    f"Initial risk factors: {initial.risk_factors}\n"
                    f"Initial comments: {[c.model_dump() for c in initial.comments]}\n"
                    f"Initial confidence: {initial.confidence:.2f}\n"
                    f"Initial confidence reasoning: {initial.confidence_reasoning}\n\n"
                    f"Reviewer Q&A:\n{qa}"
                ),
            },
        ])
    console.print(f"  [green]OK[/green] refined confidence={refined.confidence:.0%}")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="synthesize",
        confidence=refined.confidence,
        risk_level=risk_level_for(refined.confidence),
        decision="escalate",
        reason=refined.confidence_reasoning,
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"analysis": refined}


def build_graph(checkpointer):
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr),
        ("analyze", node_analyze),
        ("route", node_route),
        ("auto_approve", node_auto_approve),
        ("human_approval", node_human_approval),
        ("commit", node_commit),
        ("escalate", node_escalate),
        ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route",
        lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "commit")
    return g.compile(checkpointer=checkpointer)


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"],
            title=f"conf={payload['confidence']:.0%}",
            border_style="green",
        ))
        choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}
    return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}


async def run(pr_url: str, thread_id: str | None):
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 - SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--thread", help="Resume an existing thread")
    args = p.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()
