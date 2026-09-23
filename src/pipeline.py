import time

import ollama
from langchain_core.prompts import ChatPromptTemplate

from rag.retriever import (
    search,
    select_best_result,
    analyse_query,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "qwen3:1.7b"

RETRIEVAL_TOP_K = 10

MAX_CONTEXT_CHARS = 5000

MAX_GENERATION_TOKENS = 180


# ============================================================
# COMPATIBILITY
# ============================================================

class GenerationTimeoutError(Exception):
    pass


# ============================================================
# CONSTANTS
# ============================================================

DISCLAIMER = (
    "This information is for educational purposes only and is not "
    "a substitute for professional medical advice."
)

INSUFFICIENT_INFORMATION = (
    "Sufficient information was not found in the retrieved medical sources."
)


# ============================================================
# DISCLAIMER
# ============================================================

def ensure_disclaimer(answer):
    answer = str(answer).strip()

    if DISCLAIMER.lower() not in answer.lower():
        answer += "\n\n" + DISCLAIMER

    return answer


# ============================================================
# CONTEXT
# ============================================================

def build_context(result):
    if not result:
        return ""

    question = str(
        result.get("question", "")
    )

    answer = str(
        result.get("answer", "")
    )

    source = str(
        result.get("source", "")
    )

    url = str(
        result.get("url", "")
    )

    context = (
        f"Source question:\n"
        f"{question}\n\n"

        f"Verified medical information:\n"
        f"{answer}\n\n"

        f"Source:\n"
        f"{source}\n\n"

        f"URL:\n"
        f"{url}"
    )

    return context[:MAX_CONTEXT_CHARS]


# ============================================================
# GENERATION
# ============================================================

def generate_answer(
    question,
    context
):

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are MEDIRAG, a grounded medical information assistant.

The retrieved source has already been checked for topic and
question-type relevance.

Answer the user's question ONLY from the verified source.

Rules:
- Do not use outside knowledge.
- Do not invent medical facts.
- Do not mix diseases or unrelated topics.
- Do not diagnose the user.
- Do not give personalized medical advice.
- Do not say that information is insufficient if the verified
  source contains information relevant to the question.
- Do not add a disclaimer.
- Give a clear concise answer.
- Prefer 2 to 6 sentences.
"""
            ),
            (
                "human",
                """
Question:
{question}

Verified source:
{context}

Answer the question using only the verified source.
"""
            )
        ]
    )

    messages = prompt.format_messages(
        question=question,
        context=context
    )

    ollama_messages = []

    for message in messages:

        role = message.type

        if role == "human":
            role = "user"

        elif role == "ai":
            role = "assistant"

        elif role not in [
            "system",
            "user",
            "assistant"
        ]:
            role = "user"

        ollama_messages.append(
            {
                "role": role,
                "content": str(
                    message.content
                ),
            }
        )

    response = ollama.chat(
        model=MODEL_NAME,
        messages=ollama_messages,
        options={
            "temperature": 0,
            "num_predict": MAX_GENERATION_TOKENS,
        },
        stream=False,
        think=False,
        keep_alive="10m",
    )

    return str(
        response.get(
            "message",
            {}
        ).get(
            "content",
            ""
        )
    ).strip()


# ============================================================
# MAIN RAG PIPELINE
# ============================================================

def ask_medical_question(question):

    start_time = time.time()

    question = str(question).strip()

    if not question:

        return {
            "question": "",
            "answer": ensure_disclaimer(
                INSUFFICIENT_INFORMATION
            ),
            "results": [],
        }

    print("\n" + "=" * 70)
    print("[MEDIRAG] NEW QUESTION")
    print(question)
    print("=" * 70)

    # --------------------------------------------------------
    # RETRIEVE TOP CANDIDATES
    # --------------------------------------------------------

    retrieval_start = time.time()

    try:

        results = search(
            question,
            top_k=RETRIEVAL_TOP_K,
        )

    except Exception as e:

        print(
            "[MEDIRAG] Retrieval error:",
            str(e)
        )

        return {
            "question": question,
            "answer": ensure_disclaimer(
                INSUFFICIENT_INFORMATION
            ),
            "results": [],
        }

    print(
        f"[MEDIRAG] Retrieved {len(results)} candidates "
        f"in {time.time() - retrieval_start:.2f}s"
    )

    # --------------------------------------------------------
    # SELECT BEST MEDICAL MATCH
    # --------------------------------------------------------

    selected = select_best_result(
        question,
        results
    )

    query_type = analyse_query(
        question
    )

    # --------------------------------------------------------
    # NO VALID MATCH
    # --------------------------------------------------------

    if selected is None:

        print(
            "[MEDIRAG] No sufficiently relevant medical "
            "source found."
        )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"[MEDIRAG] Response returned in "
            f"{elapsed:.2f}s"
        )

        return {
            "question": question,
            "answer": ensure_disclaimer(
                INSUFFICIENT_INFORMATION
            ),
            "results": results,
        }

    # --------------------------------------------------------
    # SELECTED SOURCE
    # --------------------------------------------------------

    print(
        "[MEDIRAG] Selected source:"
    )

    print(
        "[MEDIRAG] Question:",
        selected.get(
            "question",
            ""
        ),
    )

    print(
        "[MEDIRAG] Source:",
        selected.get(
            "source",
            ""
        ),
    )

    print(
        "[MEDIRAG] Score:",
        round(
            float(
                selected.get(
                    "score",
                    0
                )
            ),
            4
        ),
    )

    print(
        "[MEDIRAG] Topic score:",
        selected.get(
            "topic_score",
            0
        ),
    )

    print(
        "[MEDIRAG] Type score:",
        selected.get(
            "type_score",
            0
        ),
    )

    print(
        "[MEDIRAG] Query type:",
        query_type,
    )

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context = build_context(
        selected
    )

    if not context:

        return {
            "question": question,
            "answer": ensure_disclaimer(
                INSUFFICIENT_INFORMATION
            ),
            "results": results,
        }

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    print(
        f"[MEDIRAG] Generating answer with "
        f"{MODEL_NAME}..."
    )

    generation_start = time.time()

    try:

        answer = generate_answer(
            question,
            context
        )

    except Exception as e:

        print(
            "[MEDIRAG] Qwen generation error:",
            str(e)
        )

        answer = ""

    generation_time = (
        time.time()
        - generation_start
    )

    print(
        f"[MEDIRAG] Generation completed in "
        f"{generation_time:.2f}s"
    )

    # --------------------------------------------------------
    # ROBUST FALLBACK
    #
    # Retrieval already proved that this source is relevant.
    # If Qwen fails or emits its fallback phrase, use the
    # verified source answer rather than showing a false
    # "not found" response.
    # --------------------------------------------------------

    if (
        not answer
        or "sufficient information was not found"
        in answer.lower()
    ):

        print(
            "[MEDIRAG] Using verified source answer."
        )

        answer = str(
            selected.get(
                "answer",
                INSUFFICIENT_INFORMATION
            )
        ).strip()

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    answer = ensure_disclaimer(
        answer
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"[MEDIRAG] Response returned in "
        f"{elapsed:.2f}s"
    )

    # Put selected result first for frontend source display.
    ordered_results = [
        selected
    ] + [
        r for r in results
        if r is not selected
    ]

    return {
        "question": question,
        "answer": answer,
        "results": ordered_results,
    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    tests = [
        "What are the symptoms of asthma?",
        "What are the symptoms of hypertension?",
        "How is hypertension diagnosed?",
        "What are the symptoms of diabetes?",
        "How is diabetes treated?",
        "What causes anemia?",
        "What are the symptoms of anemia?",
        "What is anemia?",
        "Tell me about lung cancer",
        "Tell about muscle cramps",
        "What causes diabetes?",
        "How can anemia be prevented?",
        "How to treat cancer?",
        "Blood cancer symptoms",
    ]

    for question in tests:

        print("\n")
        print("=" * 70)
        print("QUESTION:", question)
        print("=" * 70)

        result = ask_medical_question(
            question
        )

        print("\nANSWER")
        print("-" * 70)
        print(
            result["answer"]
        )

        print("\nSOURCE")
        print("-" * 70)

        if result["results"]:

            source = result["results"][0]

            print(
                source.get(
                    "question",
                    ""
                )
            )

            print(
                source.get(
                    "source",
                    ""
                )
            )

            print(
                source.get(
                    "url",
                    ""
                )
            )
