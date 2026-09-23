import os
import re

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "medquad_clean.csv"
)

INDEX_FILE = os.path.join(
    BASE_DIR,
    "vectorstore",
    "medquad.index"
)

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

DEFAULT_TOP_K = 10
FAISS_K = 1000


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE).fillna("")

print("MedQuAD records:", len(df))


# ============================================================
# LOAD BIOBERT
# ============================================================

print("Loading BioBERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

print("BioBERT loaded.")


# ============================================================
# LOAD FAISS
# ============================================================

index = faiss.read_index(INDEX_FILE)

print("FAISS vectors:", index.ntotal)


# ============================================================
# BASIC TEXT HELPERS
# ============================================================

def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[?!.,;:]+$", "", text)
    return text.strip()


def canonicalize_topic(topic):
    """
    Small, conservative normalization for common terminology.
    It is used only for retrieval matching.
    """

    topic = normalize_text(topic)

    topic = re.sub(
        r"^\(are\)\s*",
        "",
        topic
    )

    # Common user shorthand
    aliases = {
        "sugar": "diabetes",
        "blood pressure": "hypertension",
        "high blood pressure": "hypertension",
        "bp": "blood pressure",
    }

    return aliases.get(
        topic,
        topic
    )


# ============================================================
# INTENT DETECTION
# ============================================================

def analyse_query(query):
    q = normalize_text(query)

    # Symptoms
    if any(p in q for p in [
        "symptom",
        "symptoms",
        "signs of",
        "sign and symptoms",
        "signs and symptoms",
        "warning signs",
        "what does it feel like",
        "how does it feel",
        "indications",
    ]):
        return "symptoms"

    # Treatment
    if any(p in q for p in [
        "treatment",
        "treatments",
        "how to treat",
        "how is it treated",
        "how are they treated",
        "treated for",
        "manage",
        "management",
        "cure",
        "cured",
    ]):
        return "treatment"

    # Diagnosis
    if any(p in q for p in [
        "diagnosis",
        "diagnose",
        "diagnosed",
        "diagnostic",
        "what tests",
        "which tests",
        "tests used",
        "testing",
        "how is it diagnosed",
        "how to diagnose",
    ]):
        return "diagnosis"

    # Causes
    if any(p in q for p in [
        "what causes",
        "what cause",
        "causes of",
        "cause of",
        "caused by",
        "why does",
        "why do",
        "why is",
        "why are",
        "reason for",
        "reasons for",
    ]):
        return "causes"

    # Prevention
    if any(p in q for p in [
        "prevent",
        "prevention",
        "how to prevent",
        "how can",
        "avoid",
        "avoiding",
        "reduce the risk",
        "lower the risk",
    ]):
        return "prevention"

    # Genetic / inheritance
    if any(p in q for p in [
        "genetic",
        "genetics",
        "gene",
        "genes",
        "inherited",
        "inheritance",
        "hereditary",
        "runs in families",
        "passed down",
    ]):
        return "genetic"

    # Complications
    if any(p in q for p in [
        "complication",
        "complications",
        "long term effects",
        "long-term effects",
        "problems caused by",
    ]):
        return "complications"

    # Frequency
    if any(p in q for p in [
        "how common",
        "how many people",
        "how often",
        "incidence",
        "prevalence",
    ]):
        return "frequency"

    # General information
    if any(p in q for p in [
        "what is",
        "what are",
        "what's",
        "tell me about",
        "tell about",
        "explain",
        "define",
        "definition of",
        "information about",
        "do you have information about",
    ]):
        return "definition"

    return ""


# ============================================================
# TOPIC EXTRACTION
# ============================================================

def extract_query_topic(question):
    q = normalize_text(question)

    # ---- Symptoms first ----
    patterns = [
        r"^what are the symptoms of (.+)$",
        r"^what are the signs and symptoms of (.+)$",
        r"^what are the signs of (.+)$",
        r"^symptoms of (.+)$",
        r"^(.+)\s+symptoms$",
    ]

    for pattern in patterns:
        m = re.match(pattern, q)
        if m:
            return canonicalize_topic(m.group(1))

    # ---- Causes ----
    patterns = [
        r"^what causes (.+)$",
        r"^what are the causes of (.+)$",
        r"^causes of (.+)$",
        r"^cause of (.+)$",
    ]

    for pattern in patterns:
        m = re.match(pattern, q)
        if m:
            return canonicalize_topic(m.group(1))

    # ---- Treatment ----
    patterns = [
        r"^how to treat (.+)$",
        r"^how is (.+) treated$",
        r"^how are (.+) treated$",
        r"^treatment for (.+)$",
        r"^treatments for (.+)$",
        r"^what is the treatment for (.+)$",
        r"^what are the treatments for (.+)$",
    ]

    for pattern in patterns:
        m = re.match(pattern, q)
        if m:
            return canonicalize_topic(m.group(1))

    # ---- Diagnosis ----
    patterns = [
        r"^how is (.+) diagnosed$",
        r"^how to diagnose (.+)$",
        r"^diagnosis of (.+)$",
        r"^what tests are used for (.+)$",
        r"^which tests are used for (.+)$",
        r"^what tests diagnose (.+)$",
    ]

    for pattern in patterns:
        m = re.match(pattern, q)
        if m:
            return canonicalize_topic(m.group(1))

    # ---- Prevention ----
    patterns = [
        r"^how to prevent (.+)$",
        r"^how can (.+) be prevented$",
        r"^prevention of (.+)$",
    ]

    for pattern in patterns:
        m = re.match(pattern, q)
        if m:
            return canonicalize_topic(m.group(1))

    # ---- Genetic ----
    patterns = [
        r"^is (.+) inherited$",
        r"^what are the genetic changes related to (.+)$",
        r"^what genetic changes are related to (.+)$",
    ]

    for pattern in patterns:
        m = re.match(pattern, q)
        if m:
            return canonicalize_topic(m.group(1))

    # ---- Definition ----
    patterns = [
        r"^what is \(are\) (.+)$",
        r"^what is (.+)$",
        r"^what are (.+)$",
        r"^what's (.+)$",
        r"^tell me about (.+)$",
        r"^tell about (.+)$",
        r"^explain (.+)$",
        r"^define (.+)$",
        r"^definition of (.+)$",
        r"^information about (.+)$",
        r"^do you have information about (.+)$",
    ]

    for pattern in patterns:
        m = re.match(pattern, q)
        if m:
            topic = m.group(1).strip()
            topic = re.sub(
                r"^\(are\)\s*",
                "",
                topic
            )
            return canonicalize_topic(topic)

    # Fallback: remove common instruction words
    terms = get_medical_terms(q)

    return canonicalize_topic(
        " ".join(terms)
    )


# ============================================================
# MEDQUAD QUESTION TOPIC
# ============================================================

def extract_dataset_topic(question):
    q = normalize_text(question)

    patterns = [
        r"^what is \(are\) (.+)$",
        r"^what are the symptoms of (.+)$",
        r"^what are the signs and symptoms of (.+)$",
        r"^what causes (.+)$",
        r"^what are the causes of (.+)$",
        r"^how to prevent (.+)$",
        r"^how can (.+) be prevented$",
        r"^what are the treatments for (.+)$",
        r"^what is the treatment for (.+)$",
        r"^how is (.+) treated$",
        r"^how are (.+) treated$",
        r"^how to diagnose (.+)$",
        r"^how is (.+) diagnosed$",
        r"^what is (.+)$",
        r"^what are (.+)$",
        r"^what's (.+)$",
    ]

    for pattern in patterns:
        m = re.match(pattern, q)
        if m:
            topic = m.group(1).strip()

            topic = re.sub(
                r"^\(are\)\s*",
                "",
                topic
            )

            return canonicalize_topic(topic)

    return canonicalize_topic(q)


# ============================================================
# MEDICAL TERMS
# ============================================================

def get_medical_terms(query):
    words = re.findall(
        r"[a-zA-Z]+",
        normalize_text(query)
    )

    stop_words = {
        "what", "what's", "are", "the", "is",
        "of", "for", "a", "an", "and", "to",
        "in", "on", "with", "how", "does",
        "do", "can", "could", "would", "should",
        "tell", "me", "about", "please", "give",
        "information", "explain", "define",
        "which", "used", "use", "there",
        "any", "ways", "way", "be",
    }

    return [
        word
        for word in words
        if word not in stop_words
        and len(word) > 2
    ]


# ============================================================
# TOPIC SIMILARITY
# ============================================================

def topic_similarity(query_topic, candidate_topic):
    q = canonicalize_topic(query_topic)
    c = canonicalize_topic(candidate_topic)

    if not q or not c:
        return 0.0

    if q == c:
        return 1.0

    q_words = set(q.split())
    c_words = set(c.split())

    # Exact phrase contained in a longer specific disease
    if q in c or c in q:
        return 0.45

    if not q_words or not c_words:
        return 0.0

    overlap = len(q_words & c_words)

    # Partial overlap
    if overlap:
        ratio = overlap / max(
            len(q_words),
            len(c_words)
        )

        if ratio >= 0.75:
            return 0.45

        if ratio >= 0.5:
            return 0.25

    return 0.0


# ============================================================
# PRECOMPUTE DATASET TOPICS
# ============================================================

print("Preparing dataset topics...")

DATASET_TOPICS = df["question"].astype(str).apply(
    extract_dataset_topic
)


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

def create_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(
        dim=1
    )

    embedding = embedding.numpy().astype(
        "float32"
    )

    faiss.normalize_L2(
        embedding
    )

    return embedding


# ============================================================
# QUESTION TYPE MATCH
# ============================================================

def type_score(query_type, dataset_type):
    dataset_type = normalize_text(
        dataset_type
    )

    if not query_type:
        return 0.0

    mapping = {
        "definition": {
            "information"
        },
        "diagnosis": {
            "exams and tests"
        },
        "symptoms": {
            "symptoms"
        },
        "causes": {
            "causes"
        },
        "treatment": {
            "treatment"
        },
        "prevention": {
            "prevention"
        },
        "genetic": {
            "genetic changes",
            "inheritance"
        },
        "complications": {
            "complications"
        },
        "frequency": {
            "frequency"
        },
    }

    if dataset_type in mapping.get(
        query_type,
        set()
    ):
        return 1.0

    return 0.0


# ============================================================
# EXACT TOPIC CANDIDATES
# ============================================================

def exact_topic_candidates(query_topic):
    if not query_topic:
        return set()

    mask = DATASET_TOPICS.apply(
        lambda x: x == query_topic
    )

    return set(
        df.index[mask].tolist()
    )


# ============================================================
# LEXICAL CANDIDATES
# ============================================================

def lexical_candidates(query_terms):
    if not query_terms:
        return set()

    question_lower = (
        df["question"]
        .astype(str)
        .str.lower()
    )

    candidate_ids = set()

    for term in query_terms:

        pattern = (
            r"\b"
            + re.escape(term)
            + r"\b"
        )

        mask = question_lower.str.contains(
            pattern,
            regex=True,
            na=False
        )

        candidate_ids.update(
            df.index[mask].tolist()
        )

    return candidate_ids


# ============================================================
# SEARCH
# ============================================================

def search(
    query,
    top_k=DEFAULT_TOP_K,
    faiss_candidate_k=FAISS_K
):

    query_embedding = create_embedding(
        query
    )

    query_topic = extract_query_topic(
        query
    )

    query_type = analyse_query(
        query
    )

    query_terms = get_medical_terms(
        query
    )

    # --------------------------------------------------------
    # FAISS candidates
    # --------------------------------------------------------

    k = min(
        faiss_candidate_k,
        len(df)
    )

    scores, indices = index.search(
        query_embedding,
        k
    )

    faiss_scores = {
        int(idx): float(score)
        for score, idx in zip(
            scores[0],
            indices[0]
        )
        if idx >= 0
    }

    # --------------------------------------------------------
    # IMPORTANT:
    # Always include exact topic candidates.
    # --------------------------------------------------------

    candidate_ids = set(
        faiss_scores.keys()
    )

    candidate_ids.update(
        exact_topic_candidates(
            query_topic
        )
    )

    candidate_ids.update(
        lexical_candidates(
            query_terms
        )
    )

    results = []

    # ========================================================
    # SCORE CANDIDATES
    # ========================================================

    for idx in candidate_ids:

        if idx < 0 or idx >= len(df):
            continue

        row = df.iloc[idx]

        question = str(
            row.get("question", "")
        )

        answer = str(
            row.get("answer", "")
        )

        dataset_type = str(
            row.get("question_type", "")
        )

        candidate_topic = DATASET_TOPICS.iloc[
            idx
        ]

        topic = topic_similarity(
            query_topic,
            candidate_topic
        )

        # Question term score
        question_lower = question.lower()

        matched_terms = sum(
            1
            for term in query_terms
            if re.search(
                r"\b" + re.escape(term) + r"\b",
                question_lower
            )
        )

        question_term = (
            matched_terms
            / max(len(query_terms), 1)
        )

        # Answer term score
        answer_lower = answer.lower()

        answer_matches = sum(
            1
            for term in query_terms
            if re.search(
                r"\b" + re.escape(term) + r"\b",
                answer_lower
            )
        )

        answer_term = (
            answer_matches
            / max(len(query_terms), 1)
        )

        # Intent/type
        intent = type_score(
            query_type,
            dataset_type
        )

        # FAISS score
        faiss_score = faiss_scores.get(
            idx,
            0.0
        )

        # Exact topic
        exact_topic = (
            1.0
            if topic == 1.0
            else 0.0
        )

        # Exact type + exact topic
        exact_intent = (
            1.0
            if (
                intent == 1.0
                and topic == 1.0
            )
            else 0.0
        )

        # Definition/general-info bonus
        definition_bonus = (
            1.0
            if (
                query_type == "definition"
                and intent == 1.0
                and topic == 1.0
            )
            else 0.0
        )

        # Strong score for a fully exact medical match
        if exact_intent == 1.0:
            combined = (
                0.38 * topic
                + 0.30 * intent
                + 0.15 * question_term
                + 0.08 * faiss_score
                + 0.03 * answer_term
                + 0.04 * definition_bonus
                + 0.02 * exact_topic
            )

        else:
            combined = (
                0.34 * topic
                + 0.28 * intent
                + 0.18 * question_term
                + 0.15 * faiss_score
                + 0.03 * answer_term
                + 0.02 * exact_topic
            )

        # Strong penalty for a specific subtype when user
        # asked about the broader topic.
        if (
            topic == 0.45
            and len(query_topic.split()) <= 3
        ):
            combined -= 0.10

        results.append({
            "score": float(combined),

            "question": question,
            "answer": answer,

            "source": str(
                row.get(
                    "document_source",
                    ""
                )
            ),

            "url": str(
                row.get(
                    "document_url",
                    ""
                )
            ),

            "category": str(
                row.get(
                    "category",
                    ""
                )
            ),

            "question_type": dataset_type,

            "query_topic": query_topic,
            "candidate_topic": candidate_topic,

            "topic_score": float(topic),
            "type_score": float(intent),
            "question_term_score": float(
                question_term
            ),
            "term_score": float(
                question_term
            ),
            "answer_term_score": float(
                answer_term
            ),
            "faiss_score": float(
                faiss_score
            ),
        })

    # ========================================================
    # SORT
    # ========================================================

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# BEST VALID RESULT
# ============================================================

def select_best_result(
    query,
    results
):
    """
    Search may return several semantically similar records.
    This function selects the best medically appropriate one.
    """

    if not results:
        return None

    query_type = analyse_query(
        query
    )

    # First: exact topic + exact intent
    for result in results:

        if (
            result["topic_score"] == 1.0
            and result["type_score"] == 1.0
            and result["score"] >= 0.55
        ):
            return result

    # Second: exact topic for general questions
    if not query_type:

        for result in results:

            if (
                result["topic_score"] == 1.0
                and result["score"] >= 0.55
            ):
                return result

    # Third: exact topic with a strong semantic score.
    # Useful when dataset question_type metadata is imperfect.
    for result in results:

        if (
            result["topic_score"] == 1.0
            and result["score"] >= 0.70
        ):
            return result

    return None


# ============================================================
# TEST
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

    for q in tests:

        print("\n" + "=" * 70)
        print("QUERY:", q)
        print("=" * 70)

        results = search(
            q,
            top_k=10
        )

        selected = select_best_result(
            q,
            results
        )

        if selected:

            print("\nSELECTED:")
            print(
                selected["question"]
            )
            print(
                "Source:",
                selected["source"]
            )
            print(
                "Score:",
                round(
                    selected["score"],
                    4
                )
            )
            print(
                "Topic score:",
                selected["topic_score"]
            )
            print(
                "Type score:",
                selected["type_score"]
            )

        else:

            print(
                "\nNO VALID MEDICAL MATCH"
            )

        print("\nTOP 5:")

        for i, result in enumerate(
            results[:5],
            start=1
        ):

            print(
                f"{i}. "
                f"{result['question']} | "
                f"score={result['score']:.4f} | "
                f"topic={result['topic_score']:.2f} | "
                f"type={result['type_score']:.2f}"
            )
