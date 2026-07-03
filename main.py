from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from schemas import ChatRequest, ChatResponse, RecommendationOut
from llm import analyze_turn, compose_reply, get_client
import retrieval

def get_items_by_names(names: list[str]) -> list[dict]:
    retrieval.load_resources()
    matched = []
    for name in names:
        lower_name = name.lower()
        for r in retrieval._meta:
            if lower_name in r["name"].lower():
                matched.append(r)
                break
    return matched

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load singletons
    retrieval.load_resources()
    retrieval.load_bm25()
    get_client()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]
        analysis = analyze_turn(messages_dicts)
        
        intent = analysis.intent
        
        # Turn budget override
        if len(request.messages) >= 6 and intent == "clarify" and intent != "refuse":
            intent = "recommend"
            
        recommendations = []
        end_of_conv = False
        reply_text = ""
        
        if intent == "refuse":
            reply_out = compose_reply(
                intent,
                "User asked off-topic question or attempted prompt injection.",
                "I can only discuss SHL assessments."
            )
            reply_text = reply_out.reply
            end_of_conv = analysis.conversation_complete
            
        elif intent == "clarify":
            context = (
                f"Missing context. Role: {analysis.role_or_context}, "
                f"Seniority: {analysis.seniority}, "
                f"Skills: {analysis.must_have_skills}"
            )
            reply_out = compose_reply(
                intent,
                context,
                "Could you provide a bit more detail about the role or skills you are looking to assess?"
            )
            reply_text = reply_out.reply
            end_of_conv = False
            
        elif intent in ["recommend", "refine"]:
            query_parts = []
            if analysis.role_or_context: query_parts.append(analysis.role_or_context)
            if analysis.seniority: query_parts.append(analysis.seniority)
            if analysis.must_have_skills: query_parts.extend(analysis.must_have_skills)
            query = " ".join(query_parts) if query_parts else "assessment"
            
            filters = {}
            if analysis.preferred_test_types:
                filters["test_type_codes"] = analysis.preferred_test_types
            if analysis.max_duration_minutes is not None:
                filters["max_duration_minutes"] = analysis.max_duration_minutes
                
            results = retrieval.hybrid_search(query, top_k=10, filters=filters if filters else None)
            
            if not results and filters:
                results = retrieval.hybrid_search(query, top_k=10, filters=None)
                
            if not results:
                # Fallback to clarify
                reply_out = compose_reply(
                    "clarify",
                    "No matches found even after relaxing filters.",
                    "I couldn't find any assessments matching that. Could you adjust your requirements?"
                )
                reply_text = reply_out.reply
                end_of_conv = False
            else:
                for r in results:
                    recommendations.append(RecommendationOut(
                        name=r.get("name", ""),
                        url=r.get("url", ""),
                        test_type=r.get("test_type", "")
                    ))
                
                context_lines = []
                for r in results:
                    context_lines.append(f"- Name: {r.get('name')}, Desc: {r.get('description', '')}, Duration: {r.get('duration_minutes')}m")
                context_str = "\n".join(context_lines)
                
                reply_out = compose_reply(
                    intent,
                    context_str,
                    f"Here are {len(recommendations)} assessments that match your requirements."
                )
                reply_text = reply_out.reply
                end_of_conv = analysis.conversation_complete
                
        elif intent == "compare":
            matched_records = get_items_by_names(analysis.compare_target_names)
            unmatched = [
                name for name in analysis.compare_target_names 
                if not any(name.lower() in m["name"].lower() for m in matched_records)
            ]
            
            if not matched_records:
                reply_text = f"I couldn't find the assessments you asked to compare."
                end_of_conv = analysis.conversation_complete
            else:
                context_lines = []
                for m in matched_records:
                    context_lines.append(f"Name: {m.get('name')}\nDescription: {m.get('description', '')}\n")
                context_str = "\n".join(context_lines)
                
                reply_out = compose_reply(
                    intent,
                    context_str,
                    "Here is a comparison based on their descriptions."
                )
                reply_text = reply_out.reply
                
                if unmatched:
                    reply_text += f"\nNote: I couldn't find information for {', '.join(unmatched)}."
                    
                end_of_conv = analysis.conversation_complete

        return ChatResponse(
            reply=reply_text,
            recommendations=recommendations,
            end_of_conversation=end_of_conv
        )
    except Exception as e:
        return ChatResponse(
            reply="I'm sorry, I encountered an error while processing your request.",
            recommendations=[],
            end_of_conversation=False
        )
