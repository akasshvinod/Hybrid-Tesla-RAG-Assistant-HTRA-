from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# ============================================================
# 1. FAST + CLEAN CONTEXT FORMATTER
# ============================================================

def format_docs(docs: List[Document]) -> str:
    """
    Converts documents into a compact but fully traceable block.
    Optimized for Llama 3.1: less fluff, more structure.
    """
    out = []
    for d in docs:
        meta = d.metadata
        out.append(
            f"[SECTION]\n"
            f"Page: {meta.get('page')}\n"
            f"Chapter: {meta.get('chapter', 'Unknown')}\n"
            f"Heading: {meta.get('heading', 'Unknown')}\n"
            f"{d.page_content.strip()}"
        )
    return "\n\n".join(out)


# ============================================================
# 2. STRICT & OPTIMIZED RAG TEMPLATE (Tesla-Safe)
# ============================================================

rag_template = """
You are a **Tesla Model 3 Expert AI Assistant**.

This is a Retrieval-Augmented Generation task (RAG).
Your answer MUST come *strictly* from the provided context.
No external knowledge. No assumptions. No hallucinations.

RULES YOU MUST FOLLOW:

1. Use ONLY the information found in the context.
2. If the answer does not exist in the context:
     → Respond exactly with:
       "I don't know based on the provided manual information."
3. NEVER infer, guess, or fabricate steps.
4. If the user asks something unsafe or damaging:
     → Warn them and provide manual-approved guidance only.
5. Keep the tone technical, concise, and aligned with Tesla documentation.
6. If justification is requested, cite:
     → Page number + Section heading.

------------------------------------------------
### Previous Conversation:
{chat_history}

------------------------------------------------
### Retrieved Context (Tesla Model 3 Owner Manual):
{context_block}

------------------------------------------------
### User Question:
{query}

------------------------------------------------
### Provide the answer strictly grounded in the context:
"""


prompt_template = PromptTemplate(
    template=rag_template,
    input_variables=["query", "context_block", "chat_history"],
)


# ============================================================
# 3. BUILD FINAL PROMPT
# ============================================================

def build_rag_prompt(
    query: str,
    docs: List[Document],
    chat_history: str = "[None]"
) -> str:
    """
    Builds a strict, optimized prompt for RAG inference.
    """
    return prompt_template.format(
        query=query,
        context_block=format_docs(docs),
        chat_history=chat_history,
    )


# ============================================================
# 4. TEST HARNESS (Optional)
# ============================================================
if __name__ == "__main__":
    from src.llm.llama_client import generate_answer

    user_q = "How do I charge my Model 3?"

    mock_docs = [
        Document(
            page_content="The charging port is on the left rear side. Press and release the button to open.",
            metadata={"page": 180, "chapter": "Charging", "heading": "Opening the Charge Port"},
        ),
        Document(
            page_content="Always check your battery level and avoid dropping below 20%.",
            metadata={"page": 185, "chapter": "Charging", "heading": "Charging Best Practices"},
        ),
        Document(
            page_content="For manual cable release, follow the steps in the maintenance section.",
            metadata={"page": 184, "chapter": "Charging", "heading": "Manually Releasing Charge Cable"},
        ),
    ]

    rag_prompt = build_rag_prompt(user_q, mock_docs)
    result = generate_answer(rag_prompt)

    print("\n=== FULL PROMPT ===\n")
    print(rag_prompt)
    print("\n=== MODEL OUTPUT ===\n")
    print(result["answer"])
    print("\nLatency:", result["latency_ms"], "ms")
