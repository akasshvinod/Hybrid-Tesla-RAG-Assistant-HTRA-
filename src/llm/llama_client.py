from __future__ import annotations
import time
import requests
from typing import Dict, Any
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# ================================================
# GLOBAL: MODEL READY FLAG (prevents re-warm)
# ================================================
MODEL_READY = False


# ================================================
# 1. FAST, SAFE MODEL WARM-UP
# ================================================
def preload_model(max_retries: int = 3):
    """
    Preloads the Ollama model ONCE per interpreter session.
    Extremely fast warm-up (1 token).
    """
    global MODEL_READY

    if MODEL_READY:
        return  # already warmed up

    logger.info(f"[LLM] Warming up model: {Config.LLM_MODEL}")

    payload = {
        "model": Config.LLM_MODEL,
        "prompt": "ok",  # single-token warmup
        "stream": False,
        "options": {
            "num_predict": 1,
            "gpu_layers": -1,
        }
    }

    for attempt in range(1, max_retries + 1):
        try:
            _ = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=25
            )
            MODEL_READY = True
            logger.info(f"[LLM] Model '{Config.LLM_MODEL}' warmed and ready.")
            return

        except Exception as e:
            logger.error(f"[LLM] Warm-up attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2)
            else:
                logger.error("[LLM] Failed to warm the model after retries.")
                MODEL_READY = False
                return


# ================================================
# 2. GENERATE ANSWER (with safety + speed)
# ================================================
def generate_answer(prompt: str) -> Dict[str, Any]:
    """
    Runs optimized inference with:
        - GPU acceleration
        - 1 warm-up per session
        - JSON-safe parsing
        - latency tracking
    """
    global MODEL_READY

    if not MODEL_READY:
        preload_model()

    logger.info("[LLM] Generating answer...")

    start = time.time()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": Config.LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "gpu_layers": -1,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_ctx": 4096,
                }
            },
            timeout=300
        )

        # JSON safe-extract
        try:
            data = response.json()
            answer = data.get("response", "") or ""
        except Exception:
            logger.error("[LLM] JSON decode error")
            answer = "LLM Error: Invalid response format."

    except Exception as e:
        logger.error(f"[LLM] Runtime error: {e}")
        answer = f"LLM Error: {e}"

    end = time.time()
    latency = round((end - start) * 1000, 2)

    logger.info(f"[LLM] Completed in {latency} ms")

    return {
        "answer": answer,
        "latency_ms": latency,
    }


# ================================================
# 3. CLI TEST HARNESS
# ================================================
def main():
    print("\n🔧 **Tesla RAG - Optimized Llama Client** 🔧\n")

    while True:
        prompt = input("Prompt (empty to quit): ").strip()
        if not prompt:
            break

        result = generate_answer(prompt)
        print("\n--- OUTPUT ---\n")
        print(result["answer"])
        print("\nLatency:", result["latency_ms"], "ms\n")


if __name__ == "__main__":
    main()
