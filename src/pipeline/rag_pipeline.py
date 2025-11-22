from __future__ import annotations
from typing import Any, Dict

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableLambda

from src.retrieval.retriever import safe_retrieve
from src.llm.prompts import build_rag_prompt
from src.llm.llama_client import generate_answer
from src.utils.logger import get_logger

logger = get_logger(__name__)

# =========================================================
# CHAT MEMORY (Optimized)
# =========================================================

MAX_TURNS = 10   # store only 10 turns → prevents prompt bloat
chat_history = ChatMessageHistory()

def format_history() -> str:
    """Compact + fast chat history formatter."""
    msgs = chat_history.messages

    # Trim to last N turns (20 messages)
    if len(msgs) > MAX_TURNS * 2:
        chat_history.messages = msgs[-MAX_TURNS * 2:]

    if not chat_history.messages:
        return "[None]"

    lines = []
    for m in chat_history.messages:
        if m.type == "human":
            lines.append(f"[User] {m.content}")
        else:
            lines.append(f"[AI] {m.content}")

    return "\n".join(lines)


# =========================================================
# AUTO CHAPTER DETECTION (Lightweight)
# =========================================================

KNOWN_CHAPTERS = {
    "charging", "safety", "autopilot", "driving", "interior", "exterior",
    "maintenance", "controls", "specifications", "warning"
}

def detect_chapter_from_query(query: str) -> str | None:
    q = query.lower()
    for ch in KNOWN_CHAPTERS:
        if ch in q:
            return ch.capitalize()
    return None


# =========================================================
# PIPELINE PHASE 1: RETRIEVAL
# =========================================================

def run_retriever(input: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Running retrieval...")

    question = input["question"]
    chapter = input.get("chapter") or detect_chapter_from_query(question)
    input["chapter"] = chapter

    result = safe_retrieve(question, chapter=chapter)

    input["docs"] = result["docs"]
    input["no_answer"] = result["no_answer"]
    input["retrieval_latency"] = result["latency_ms"]

    # Context safety: reject weak matches
    if not result["docs"] or sum(len(d.page_content) for d in result["docs"]) < 120:
        input["no_answer"] = True

    return input


# =========================================================
# PIPELINE PHASE 2: PROMPT BUILDING
# =========================================================

def run_prompt(input: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Building prompt...")

    if input["no_answer"]:
        input["prompt"] = "I don't know based on the provided manual information."
        return input

    input["prompt"] = build_rag_prompt(
        query=input["question"],
        docs=input["docs"],
        chat_history=format_history()
    )
    return input


# =========================================================
# PIPELINE PHASE 3: LLM INFERENCE
# =========================================================

def run_llm(input: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Running Llama 3.1 inference...")

    llm_out = generate_answer(input["prompt"])
    input["answer"] = llm_out["answer"]
    input["llm_latency"] = llm_out["latency_ms"]

    return input


# =========================================================
# PIPELINE PHASE 4: MEMORY UPDATE
# =========================================================

def update_history(input: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Updating chat memory...")

    chat_history.add_user_message(input["question"])
    chat_history.add_ai_message(input["answer"])

    return input


# =========================================================
# FINAL LCEL PIPELINE
# =========================================================

# Cleaner & faster: no Passthrough / no extra layers
RAG_PIPELINE = (
      RunnableLambda(run_retriever)
    | RunnableLambda(run_prompt)
    | RunnableLambda(run_llm)
    | RunnableLambda(update_history)
)


# =========================================================
# PUBLIC API — call this from CLI, FastAPI, Streamlit, etc.
# =========================================================

def answer_query(question: str, chapter: str = None) -> Dict[str, Any]:
    logger.info(f"Processing query: {question}")

    result = RAG_PIPELINE.invoke({
        "question": question,
        "chapter": chapter
    })

    return {
        "answer":     result["answer"],
        "retrieval_latency": result.get("retrieval_latency"),
        "llm_latency":       result.get("llm_latency"),
        "total_latency":     (result.get("retrieval_latency", 0) +
                              result.get("llm_latency", 0)),
        "chapter_used":      result.get("chapter"),
        "docs_used":         len(result.get("docs", [])),
        "chat_history":      format_history(),
    }


# =========================================================
# CLI TEST
# =========================================================

if __name__ == "__main__":
    res = answer_query("How do I charge my Model 3?")
    print("\n=== ANSWER ===\n", res["answer"])
    print("\n=== METRICS ===\n", res)

    res2 = answer_query("What if the cable is stuck?")
    print("\n=== FOLLOW-UP ===\n", res2["answer"])
    print("\n=== HISTORY ===\n", res2["chat_history"])
