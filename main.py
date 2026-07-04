from fastapi import FastAPI
from contextlib import asynccontextmanager
from schemas import ChatRequest, ChatResponse, RecommendationOut
from llm import analyze_turn, compose_reply, get_client
import retrieval

# ---------------------------------------------------------------------------
# Default query used when the turn-budget forces a recommend on a vague history
# ---------------------------------------------------------------------------
_BUDGET_FALLBACK_QUERY = (
    "general cognitive ability verbal numerical reasoning personality workplace"
)

# ---------------------------------------------------------------------------
# Helper: look up catalog items by name (case-insensitive substring)
# ---------------------------------------------------------------------------

def get_items_by_names(names: list[str]) -> list[dict]:
    retrieval.load_resources()
    matched: list[dict] = []
    seen: set[str] = set()
    for name in names:
        lower_name = name.lower()
        for r in retrieval._meta:
            rname = r["name"]
            if rname in seen:
                continue
            if lower_name in rname.lower():
                matched.append(r)
                seen.add(rname)
                break
    return matched


# ---------------------------------------------------------------------------
# Helper: extract previously recommended items from conversation history
# ---------------------------------------------------------------------------

def extract_previous_recommendations(messages_dicts: list[dict]) -> list[dict]:
    """Find catalog items whose names appear in the most recent assistant
    shortlist message.  Walks backwards through assistant turns so the
    latest shortlist wins."""
    retrieval.load_resources()
    for msg in reversed(messages_dicts):
        if msg["role"] != "assistant":
            continue
        content = msg["content"]
        found = [r.copy() for r in retrieval._meta if r["name"] in content]
        if found:
            return found
    return []


# ---------------------------------------------------------------------------
# Lifespan: warm all singletons at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    retrieval.load_resources()
    retrieval.load_bm25()
    get_client()
    yield


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]

        # ── First pass: classify without catalog hint to determine intent ────
        analysis = analyze_turn(messages_dicts)

        intent = analysis.intent

        # ── Turn-budget override ─────────────────────────────────────────────
        # If we are at ≥6 messages, still getting 'clarify', and it isn't a
        # refuse/compare, force a recommend so we never end with empty results.
        if len(request.messages) >= 6 and intent == "clarify":
            intent = "recommend"

        recommendations: list[RecommendationOut] = []
        end_of_conv = False
        reply_text = ""

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # REFUSE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if intent == "refuse":
            reply_out = compose_reply(
                "refuse",
                "User asked an off-topic question or attempted prompt injection.",
                "I'm sorry, I can only assist with SHL assessment recommendations.",
            )
            reply_text = reply_out.reply
            end_of_conv = False  # refuse never ends the conversation cleanly

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CLARIFY
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        elif intent == "clarify":
            context = (
                f"Missing context. Role: {analysis.role_or_context}, "
                f"Seniority: {analysis.seniority}, "
                f"Skills: {analysis.must_have_skills}"
            )
            reply_out = compose_reply(
                "clarify",
                context,
                "Could you provide a bit more detail about the role or skills you are looking to assess?",
            )
            reply_text = reply_out.reply
            end_of_conv = False

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # RECOMMEND / REFINE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        elif intent in ("recommend", "refine"):
            # Build query from accumulated profile fields
            query_parts: list[str] = []
            if analysis.role_or_context:
                query_parts.append(analysis.role_or_context)
            if analysis.seniority:
                query_parts.append(analysis.seniority)
            if analysis.must_have_skills:
                query_parts.extend(analysis.must_have_skills)

            # Turn-budget fallback: if query is still empty/sparse, use a
            # broad default so search returns something sensible
            query = " ".join(query_parts).strip() if query_parts else _BUDGET_FALLBACK_QUERY

            # Build filters (omit key entirely if source is None/empty)
            filters: dict = {}

            if analysis.preferred_test_types:
                filters["test_type_codes"] = analysis.preferred_test_types
            if analysis.max_duration_minutes is not None:
                filters["max_duration_minutes"] = analysis.max_duration_minutes

            results = retrieval.hybrid_search(
                query, top_k=10, filters=filters if filters else None
            )
            
            # Fallback: relax filters once if empty
            if not results and filters:
                results = retrieval.hybrid_search(query, top_k=10, filters=None)

            # ── Refine-specific: merge previous recs + focused new search ──
            if intent == "refine":
                prev_records = extract_previous_recommendations(messages_dicts)

                # Focused search on last user message (the refinement request)
                last_user_content = ""
                for msg in reversed(messages_dicts):
                    if msg["role"] == "user":
                        last_user_content = msg["content"]
                        break
                focused_results: list[dict] = []
                if last_user_content:
                    focused_results = retrieval.hybrid_search(
                        last_user_content, top_k=5, filters=None
                    )

                # Merge: prev_records first (guaranteed), then focused
                # (new items), then broad results — deduplicate by name
                seen_names: set[str] = set()
                merged: list[dict] = []
                for r in prev_records + focused_results + results:
                    name = r["name"]
                    if name not in seen_names:
                        merged.append(r)
                        seen_names.add(name)
                results = merged[:10]

            # If still empty → degrade to clarify
            if not results:
                reply_out = compose_reply(
                    "clarify",
                    "No matches found even after relaxing filters.",
                    "I couldn't find any matching assessments. Could you adjust your requirements?",
                )
                reply_text = reply_out.reply
                end_of_conv = False
            else:
                # ── Second LLM pass: re-classify WITH catalog hint ──────────
                # This lets the model populate `selected_assessment_names` so
                # the structured array reflects what the reply text discusses.
                catalog_names = [r["name"] for r in results]
                analysis2 = analyze_turn(messages_dicts, catalog_names=catalog_names)

                # Use selected_assessment_names from second pass to ground the array
                selected_names = analysis2.selected_assessment_names

                # Build a name→record lookup for O(1) access
                record_by_name: dict[str, dict] = {r["name"]: r for r in results}

                # If the LLM returned valid selections, use those; otherwise
                # fall back to all retrieved results (safety net)
                if selected_names:
                    grounded_records = [
                        record_by_name[n]
                        for n in selected_names
                        if n in record_by_name
                    ]
                    # Ensure we always have at least 1 result
                    if not grounded_records:
                        grounded_records = results
                else:
                    grounded_records = results

                # Build structured recommendations from grounded catalog records
                for r in grounded_records:
                    recommendations.append(
                        RecommendationOut(
                            name=r.get("name", ""),
                            url=r.get("url", ""),
                            test_type=r.get("test_type", ""),
                        )
                    )

                # Build context for compose_reply using only grounded records
                context_lines = [
                    f"- Name: {r['name']}, "
                    f"Description: {r.get('description', '')}, "
                    f"Duration: {r.get('duration_minutes')}m"
                    for r in grounded_records
                ]
                context_str = "\n".join(context_lines)

                reply_out = compose_reply(
                    intent,
                    context_str,
                    f"Here are {len(recommendations)} assessments that match your requirements.",
                )
                reply_text = reply_out.reply
                # Use the more-informed second-pass for conversation_complete
                end_of_conv = analysis2.conversation_complete

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # COMPARE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        elif intent == "compare":
            target_names = analysis.compare_target_names
            matched_records = get_items_by_names(target_names)

            # Names that could NOT be matched in the catalog
            matched_lower = {m["name"].lower() for m in matched_records}
            unmatched = [
                name
                for name in target_names
                if not any(name.lower() in ml for ml in matched_lower)
            ]

            if not matched_records:
                reply_text = "I couldn't find the assessments you asked to compare in the catalog."
                end_of_conv = analysis.conversation_complete
            else:
                context_parts = [
                    f"Name: {m['name']}\nDescription: {m.get('description', '')}"
                    for m in matched_records
                ]
                context_str = "\n\n".join(context_parts)

                reply_out = compose_reply(
                    "compare",
                    context_str,
                    "Here is a comparison based on their catalog descriptions.",
                )
                reply_text = reply_out.reply
                if unmatched:
                    reply_text += (
                        f"\n\nNote: I couldn't find catalog entries for: "
                        + ", ".join(unmatched)
                        + "."
                    )
                end_of_conv = analysis.conversation_complete

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CONFIRM
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        elif intent == "confirm":
            # Re-populate the final shortlist from conversation history
            prev_records = extract_previous_recommendations(messages_dicts)

            # Also honour any names the LLM extracted via selected_assessment_names
            if not prev_records and analysis.selected_assessment_names:
                prev_records = get_items_by_names(analysis.selected_assessment_names)

            for r in prev_records:
                recommendations.append(
                    RecommendationOut(
                        name=r.get("name", ""),
                        url=r.get("url", ""),
                        test_type=r.get("test_type", ""),
                    )
                )

            context_str = (
                "Final shortlist: " + ", ".join(r["name"] for r in prev_records)
                if prev_records
                else "No assessments found in conversation history."
            )
            reply_out = compose_reply(
                "confirm",
                context_str,
                "Great! Your shortlist has been confirmed.",
            )
            reply_text = reply_out.reply
            end_of_conv = True

        # ── Safety net: ensure reply is never empty ──────────────────────────
        if not reply_text:
            reply_text = "I'm sorry, something went wrong. Please try again."

        return ChatResponse(
            reply=reply_text,
            recommendations=recommendations,
            end_of_conversation=end_of_conv,
        )

    except Exception:
        # Last-resort guard: never return a non-schema response or raw 500
        return ChatResponse(
            reply="I'm sorry, I encountered an unexpected error. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )
