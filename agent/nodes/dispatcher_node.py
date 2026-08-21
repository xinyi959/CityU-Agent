"""Dispatcher: fan out one retriever call per router sub-decision.

Replaces the graph-level conditional edges (router -> one retriever). The
router's ``decisions`` may contain several sub-questions (compound query),
each with a ``field`` (the dispatcher derives ``retrieval_type`` from it
and falls back to ``_fallback_decision_list`` when the plan is missing);
the dispatcher loops over them, runs the matching retriever node with a
per-decision sub-state, and merges the Evidence into a single list.

Per-decision sub-state:

- ``query``        <- dec.sub_query (NOT the full compound query: the
                       section retriever vector-searches on this text, and
                       the other sub-questions' tokens would pollute it)
- ``field``        <- dec.field
- ``programme_ref``<- dec.programme_ref if set (cross-programme compound),
                       else the top-level programme_ref shared by the turn
- ``programme_ids``<- (Plan A) the programme ids a preceding summary
                       decision resolved to, when THIS decision has no
                       programme referent of its own. Scopes the metadata /
                       section retrieval to the recommended set instead of a
                       whole-corpus semantic search ("which programme should
                       I apply for? and how much should I pay?").
- messages / resolved_programme_ref flow through unchanged so the
  multi-turn referent fallback chain keeps working.

Dependency ordering: decisions are processed in order, and only a summary
that appears BEFORE a referent-less metadata/section decision can seed its
scope (forward propagation). The natural wording "recommend X, then tell me
its fee" keeps the recommendation first, so this covers the reported case.

Evidence is deduplicated by id (two sub-questions can resolve to the same
field, e.g. "fee" and "cost" both -> P53-tuition_fee).

Safety: a state without ``decisions`` (direct invoke, pre-Phase-1
checkpoint) falls back to the router's rule-based plan on the full query.
"""

from agent.nodes.metadata_retriever_node import metadata_retriever_node
from agent.nodes.router_node import _fallback_decision_list
from agent.nodes.section_retriever_node import section_retriever_node
from agent.nodes.summary_retriever_node import summary_retriever_node
from agent.state.router_schema import retrieval_type_of

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

    # Programme ids collected from summary (recommendation) decisions earlier
    # in this turn. A later metadata/section sub-question with no programme
    # referent of its own ("which programme should I apply for? and how much
    # should I pay?") is scoped to this set instead of falling back to a
    # whole-corpus semantic search.
    scope_ids = []

    for dec in decisions:
        rtype = dec.get("retrieval_type") or retrieval_type_of(dec.get("field"))
        retriever = RETRIEVER_NODES.get(rtype)
        if retriever is None:
            continue

        dec_ref = dec.get("programme_ref")
        top_ref = state.get("programme_ref")

        sub_state = {
            **state,
            "query": dec.get("sub_query") or state.get("query", ""),
            "field": dec.get("field"),
            "programme_ref": dec_ref or top_ref,
        }

        # Scope propagation (Plan A): hand the recommended set to the
        # retriever. The retrievers already resolve an explicit single
        # programme first (router ref -> resolved ref -> query text ->
        # messages); only when that fails do they fall back to this set. So
        # no need to re-derive "has an explicit referent" here -- a ref that
        # does not resolve to a real programme (e.g. the top-level ref
        # {"programme_name": "AI"} for a recommendation) correctly lets the
        # scope take over.
        if rtype in ("metadata", "section") and scope_ids:
            sub_state["programme_ids"] = list(scope_ids)

        out = retriever(sub_state)

        # Recommendation decisions seed the scope for subsequent sub-questions.
        if rtype == "summary":
            for e in out.get("evidence", []):
                if e.programme_id and e.programme_id not in scope_ids:
                    scope_ids.append(e.programme_id)

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
