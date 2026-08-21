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
- ``programme_ids``<- the programme ids a preceding summary decision
                       resolved to THIS turn, or the set persisted by the
                       PREVIOUS turn. Scopes a referent-less metadata /
                       section sub-question to that set instead of a
                       whole-corpus semantic search ("which programme should
                       I apply for? and how much should I pay?" / "Any apply
                       requirement I should fulfill?").
- ``resolved_programme_refs`` <- persisted programme set for the NEXT turn
                       (the recommendation set if one happened this turn,
                       else the resolved single/multi refs). List of
                       {programme_id, programme_name}.
- messages flow through unchanged so the multi-turn referent fallback
  chain keeps working.

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
from rag.programme_resolver import get_programmes

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

    # Programme set persisted by the previous turn, reused as scope when this
    # turn's sub-question omits a referent ("Any apply requirement I should
    # fulfill?").
    persisted_ids = [
        ref.get("programme_id")
        for ref in (state.get("resolved_programme_refs") or [])
        if ref and ref.get("programme_id")
    ]

    # Programme ids collected from summary (recommendation) decisions this
    # turn, scoped to later referent-less metadata/section decisions.
    scope_ids = []

    # Programme ids resolved by the retrievers this turn (single refs or the
    # scoped set), persisted when no recommendation happened.
    resolved_ids = []

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

        # Scope: this turn's recommendation set wins; otherwise the set
        # persisted by the previous turn. The retrievers resolve an explicit
        # single programme first (router ref -> resolved refs -> query text ->
        # messages); only when that fails do they fall back to this set.
        scope = list(scope_ids) or persisted_ids
        if rtype in ("metadata", "section") and scope:
            sub_state["programme_ids"] = scope

        out = retriever(sub_state)

        # Recommendation decisions seed the scope for subsequent sub-questions.
        if rtype == "summary":
            for e in out.get("evidence", []):
                if e.programme_id and e.programme_id not in scope_ids:
                    scope_ids.append(e.programme_id)

        for ref in out.get("resolved_programme_refs") or []:
            pid = (ref or {}).get("programme_id")
            if pid and pid not in resolved_ids:
                resolved_ids.append(pid)

        for e in out.get("evidence", []):
            if e.id in seen_ids:
                continue
            seen_ids.add(e.id)
            evidence.append(e)

    # Persist the programme set for the NEXT turn. The recommendation set
    # wins (most recent subject); otherwise the resolved single/multi refs
    # from the retrievers; otherwise keep the previous value (omit the key).
    if scope_ids:
        ids = list(scope_ids)
    elif resolved_ids:
        ids = resolved_ids
    else:
        ids = None

    result = {"evidence": evidence}
    if ids is not None:
        name_map = {
            p["programme_id"]: p.get("name") for p in get_programmes()
        }
        result["resolved_programme_refs"] = [
            {"programme_id": pid, "programme_name": name_map.get(pid)}
            for pid in ids
        ]
    return result
