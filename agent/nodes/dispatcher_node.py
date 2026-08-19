"""Dispatcher: fan out one retriever call per router sub-decision.

Replaces the graph-level conditional edges (router -> one retriever). The
router's ``decisions`` may contain several sub-questions (compound query),
each with its own retrieval_type / field / sub_query; the dispatcher loops
over them, runs the matching retriever node with a per-decision sub-state,
and merges the Evidence into a single list.

Per-decision sub-state:

- ``query``        <- dec.sub_query (NOT the full compound query: the
                       section retriever vector-searches on this text, and
                       the other sub-questions' tokens would pollute it)
- ``field``        <- dec.field
- ``programme_ref``<- dec.programme_ref if set (cross-programme compound),
                       else the top-level programme_ref shared by the turn
- messages / resolved_programme_ref flow through unchanged so the
  multi-turn referent fallback chain keeps working.

Evidence is deduplicated by id (two sub-questions can resolve to the same
field, e.g. "fee" and "cost" both -> P53-tuition_fee).

Safety: a state without ``decisions`` (direct invoke, pre-Phase-1
checkpoint) falls back to the router's rule-based plan on the full query.
"""

from agent.nodes.metadata_retriever_node import metadata_retriever_node
from agent.nodes.router_node import _fallback_decision_list
from agent.nodes.section_retriever_node import section_retriever_node
from agent.nodes.summary_retriever_node import summary_retriever_node

RETRIEVER_NODES = {
    "metadata": metadata_retriever_node,
    "section": section_retriever_node,
    "summary": summary_retriever_node,
}


def dispatcher_node(state):
    decisions = state.get("decisions") or []

    if not decisions:
        plan = _fallback_decision_list(state.get("query") or "")
        decisions = [d.model_dump() for d in plan.decisions]

    evidence = []
    seen_ids = set()
    resolved_ref = state.get("resolved_programme_ref")

    for dec in decisions:
        retriever = RETRIEVER_NODES.get(dec.get("retrieval_type"))
        if retriever is None:
            continue

        sub_state = {
            **state,
            "query": dec.get("sub_query") or state.get("query", ""),
            "field": dec.get("field"),
            "programme_ref": (
                dec.get("programme_ref") or state.get("programme_ref")
            ),
        }

        out = retriever(sub_state)

        for e in out.get("evidence", []):
            if e.id in seen_ids:
                continue
            seen_ids.add(e.id)
            evidence.append(e)

        if not resolved_ref and out.get("resolved_programme_ref"):
            resolved_ref = out["resolved_programme_ref"]

    return {
        "evidence": evidence,
        "resolved_programme_ref": resolved_ref,
    }
