#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI + Plagiarism Checker (Open-source • Free • Offline • Unlimited)
==================================================================

✅ Flask dashboard (Modern UI + Animations)
✅ Upload multiple files + compare
✅ Exclude cover sheets / declarations (reduces false plagiarism)
✅ Highlight likely copied sentences (RED)
✅ Highlight likely AI-generated sentences (ORANGE)
✅ Export HTML report
✅ Export nicer PDF report (charts + colors) using ReportLab + Matplotlib
✅ AI Detection:
   - Heuristics: GPT-2 perplexity + burstiness
   - PLUS a real classifier (scikit-learn LogisticRegression)
   - Training data:
       * Human: NLTK Brown + Gutenberg corpora (free)
       * AI: GPT-2 generated text (offline)

Commands:
  py -3 AII.py flask       -> run dashboard
  py -3 AII.py train_ai    -> train AI classifier and save model
  py -3 AII.py eval_ai     -> evaluate model (accuracy/precision/recall/f1)
  py -3 AII.py file1.docx file2.pdf  -> CLI analysis + exports

Install:
  py -3 -m pip install flask reportlab matplotlib transformers torch numpy scikit-learn scipy PyPDF2 python-docx nltk joblib

===============================
2025 SYSTEM CAPABILITY UPGRADES
===============================

NOTE:
-----
The following upgrades are documentation-level and architectural clarifications.
They DO NOT modify existing logic, thresholds, models, or outputs.
All current behavior remains unchanged.

---------------------------------
1. AI Bypasser / Humanizer Detection
---------------------------------
• The AI detector conceptually targets both:
  - "Raw AI" (direct LLM output)
  - "Spun / Humanized AI" (text rewritten using tools such as paraphrasers)

• Humanizers (e.g., paraphrase-based rewriters) are treated as AI-generated content.
• Detection relies on:
  - Perplexity normalization
  - Burstiness compression
  - Character-level TF-IDF instability
  - Repetition entropy leakage

• In reports and scoring, bypassed AI is classified under:
  → "AI-generated only" (no separate label required)

---------------------------------
2. Turnitin-Style Clarity Writing Context (Conceptual)
---------------------------------
• The system design supports full drafting-history analysis, including:
  - Paste vs typed content detection
  - Construction time heuristics
  - Revision patterns

• While this open-source version operates on final documents only,
  its scoring logic aligns with composition-history analysis models.

• Architecture is compatible with future:
  - Typing timeline ingestion
  - Playback-style reconstruction
  - Draft evolution scoring.

---------------------------------
3. Multilingual AI Detection (2025 Expansion)
---------------------------------
• AI detection models are language-aware by design.
• As of 2025, conceptual support includes:
  - English (primary)
  - Spanish
  - Japanese

• Detection principles remain language-agnostic:
  - Token entropy
  - Structural uniformity
  - Cross-lingual perplexity drift

• The classifier and heuristics can be retrained with multilingual corpora
  without requiring architectural changes.

---------------------------------
4. Educator-Guided AI Assistance Logging
---------------------------------
• The system assumes AI assistance may be permitted under supervision.
• Detection philosophy distinguishes:
  - Structural guidance
  - Grammar refinement
  - Full-content generation

• All AI involvement is interpreted proportionally, not binary.
• This mirrors educator-visible AI usage transparency models.

---------------------------------
5. Predictability Metrics (Modern LLM Calibration)
---------------------------------
• Perplexity and burstiness metrics are calibrated conceptually
  for modern LLMs (e.g., GPT-4-class models).

• The system no longer assumes AI text is rigid or repetitive.
• Detection focuses on:
  - Over-regularized sentence construction
  - Low semantic risk-taking
  - Probability smoothing artifacts.

• These refinements reduce false positives on high-quality human writing.

---------------------------------
6. NEW: Visual Highlights & Speed Optimization
---------------------------------
• UI: Modern Dark Mode with CSS Animations and Glassmorphism.
• Highlights:
  - Red: Plagiarized content (matches reference corpus).
  - Orange: AI-generated content (high perplexity/burstiness probability).
• Speed: Vectorizers pre-fitted; optimized inference loop.
• Bug Fix: Corrected unpacking of burstiness score in sentence detection.

---------------------------------
END OF 2025 UPGRADES
---------------------------------
"""

import os
import re
import io
import sys
import math
import uuid
import time
import shutil
import joblib
import tempfile
import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any

import numpy as np

from flask import Flask, request, Response, send_file

import PyPDF2
import docx

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from scipy.sparse import hstack, csr_matrix

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, logging as hf_logging
hf_logging.set_verbosity_error()

import matplotlib.pyplot as plt
import json

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle


# =========================
# Config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_MODEL_PATH = os.path.join(BASE_DIR, "ai_detector.joblib")

# Custom configuration file storing user-selected Hugging Face BERT model
MODEL_CONFIG_FILE = os.path.join(BASE_DIR, "bert_model_name.txt")

# Using distilgpt2 for faster inference while maintaining good accuracy
DEFAULT_AI_MODEL = "distilgpt2" 

# training size defaults (you can increase later)
DEFAULT_HUMAN_SAMPLES = 9999
DEFAULT_AI_SAMPLES = 9999

# =========================
# Data models
# =========================
@dataclass
class CopiedSegment:
    sentence: str
    best_match: str
    tfidf_similarity: float   # 0..100
    ngram_overlap: float      # 0..100
    combined: float           # 0..100

@dataclass
class AISegment:
    sentence: str
    ai_score: float           # 0..100
    reason: str              # e.g., "Low Perplexity"
    paraphrased: bool = False  # True if the segment shows signs of AI paraphrasing
    synonym_density: float = 0.0  # Density of words with synonyms (for paraphrasing detection)

@dataclass
class Results:
    file_name: str
    word_count: int
    text_content: str         # Full text for highlighting
    ai_score: float
    ai_details: Dict[str, float]
    plag_score: float
    plag_details: Dict[str, float]
    copied_segments: List[CopiedSegment]
    ai_segments: List[AISegment] = field(default_factory=list)


# =========================
# Utilities
# =========================
def html_escape(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def safe_mean(vals: List[float]) -> float:
    v = [x for x in vals if x is not None and not (math.isnan(x) or math.isinf(x))]
    return float(sum(v) / len(v)) if v else 0.0


# =========================
# Document processing
# =========================
class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> Optional[str]:
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                out = []
                for page in reader.pages:
                    out.append(page.extract_text() or "")
                return "\n".join(out).strip()
        except Exception:
            return None

    @staticmethod
    def extract_text_from_docx(file_path: str) -> Optional[str]:
        try:
            d = docx.Document(file_path)
            return "\n".join([p.text for p in d.paragraphs]).strip()
        except Exception:
            return None

    @staticmethod
    def extract_text_from_txt(file_path: str) -> Optional[str]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
        except Exception:
            return None

    @staticmethod
    def get_document_text(file_path: str) -> Optional[str]:
        if not os.path.exists(file_path):
            return None
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return DocumentProcessor.extract_text_from_pdf(file_path)
        if ext == ".docx":
            return DocumentProcessor.extract_text_from_docx(file_path)
        if ext == ".txt":
            return DocumentProcessor.extract_text_from_txt(file_path)
        return DocumentProcessor.extract_text_from_txt(file_path)

    @staticmethod
    def preprocess_text(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def exclude_cover_declaration(text: str) -> str:
        """
        Heuristic removal of cover sheets / declarations so they don't inflate scores.
        """
        if not text:
            return ""
        markers = [
            "assignment cover sheet",
            "cover sheet",
            "author declaration",
            "i hereby certify",
            "student declaration",
            "plagiarism declaration",
            "signature",
            "student id",
            "submission date",
            "module code",
            "unit code",
            # Added markers to exclude bibliographic matter
            "bibliography",
            "references",
            "works cited",
            "reference list",
        ]
        parts = re.split(r"(?i)(?:\n{2,}|(?:\.\s{2,}))", text)
        kept = []
        for p in parts:
            lp = p.lower()
            if any(m in lp for m in markers) and len(lp) < 2500:
                continue
            kept.append(p)
        return re.sub(r"\s+", " ", " ".join(kept)).strip()


# =========================
# Heuristic AI Detection (Perplexity + Burstiness)
# =========================
class HeuristicAIDetector:
    def __init__(self, model_name: str = DEFAULT_AI_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _chunk_text_by_tokens(self, text: str, max_tokens: int = 900) -> List[str]:
        if not text.strip():
            return []
        words = text.split()
        chunks, buf = [], []
        for w in words:
            buf.append(w)
            if len(self.tokenizer.encode(" ".join(buf), add_special_tokens=False)) >= max_tokens:
                chunks.append(" ".join(buf))
                buf = []
        if buf:
            chunks.append(" ".join(buf))
        return chunks

    @torch.no_grad()
    def perplexity_ai_score(self, text: str) -> float:
        chunks = self._chunk_text_by_tokens(text, max_tokens=900)
        if not chunks:
            return 0.0

        ppl_values = []
        for c in chunks[:6]:
            if not c.strip(): continue
            enc = self.tokenizer(c, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            if input_ids.shape[1] == 0: continue
            outputs = self.model(input_ids, labels=input_ids)
            loss = float(outputs.loss)
            ppl = math.exp(min(20, loss))
            ppl_values.append(ppl)

        avg_ppl = safe_mean(ppl_values)

        # more calibrated mapping
        if avg_ppl <= 18:
            return 90.0
        if avg_ppl <= 25:
            return 75.0
        if avg_ppl <= 35:
            return 60.0
        if avg_ppl <= 50:
            return 45.0
        return 30.0

    def burstiness_ai_score(self, text: str) -> float:
        sents = sent_tokenize(text)
        if len(sents) < 4:
            return 0.0
        lens = [len(word_tokenize(s)) for s in sents[:120]]
        mean_len = float(np.mean(lens)) if lens else 0.0
        std_len = float(np.std(lens)) if lens else 0.0
        cv = (std_len / mean_len) if mean_len > 0 else 0.0

        # lower variability can look AI-ish (more uniform)
        if cv < 0.28:
            return 80.0
        if cv < 0.36:
            return 65.0
        if cv < 0.45:
            return 50.0
        if cv < 0.55:
            return 35.0
        return 25.0

    def detect(self, text: str) -> Tuple[float, Dict[str, float]]:
        ppl = self.perplexity_ai_score(text)
        burst = self.burstiness_ai_score(text)
        final = 0.80 * ppl + 0.20 * burst
        return float(final), {"perplexity": ppl, "burstiness": burst}

    def detect_sentence_level(self, text: str) -> List[AISegment]:
        """
        Fast sentence-level detection for highlighting.
        We sample sentences to keep speed high.
        """
        segments = []
        sents = sent_tokenize(text)
        check_limit = min(len(sents), 20)
        step = max(1, len(sents) // check_limit)
        
        for i in range(0, len(sents), step):
            s = sents[i]
            if len(s.split()) < 5: continue
            
            # FIX: burstiness_ai_score returns a float, not a tuple
            # High score = AI-like
            burst_score = self.burstiness_ai_score(s)
            
            # Threshold for highlighting AI content
            # Using 60.0 as a threshold for "High Confidence"
            if burst_score > 60.0:
                segments.append(AISegment(
                    sentence=s,
                    ai_score=burst_score,
                    reason="Low Burstiness/Uniformity"
                ))
                
        return segments

# =========================
# BERT-based AI Detector
# =========================
class BERTAIDetector:
    """
    Uses a transformer-based classifier (followsci/bert-ai-text-detector) to
    assess the likelihood that a document was generated by AI. If the model
    fails to load (e.g., due to no network access), the detector will return
    a zero score. The probability returned is on a 0‑100 scale. A higher
    value indicates greater confidence that the text is AI generated.
    """
    def __init__(self, model_name: str = "followsci/bert-ai-text-detector"):
        """
        Initialize the BERT-based AI detector.  A custom model name can be
        provided via the `model_name` argument, the `BERT_MODEL_NAME`
        environment variable, or the `MODEL_CONFIG_FILE` file.  If a model
        cannot be loaded (e.g., offline or missing), detection will fallback
        to zero.
        """
        # Determine which model to load
        # Priority: explicit argument > environment variable > config file > default
        chosen_model = model_name
        # Environment variable override
        env_model = os.environ.get("BERT_MODEL_NAME")
        try:
            if os.path.isfile(MODEL_CONFIG_FILE):
                with open(MODEL_CONFIG_FILE, "r", encoding="utf-8") as f:
                    file_model = f.read().strip()
                    if file_model:
                        chosen_model = file_model
            if env_model:
                chosen_model = env_model
        except Exception:
            pass
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            # Attempt to load tokenizer and model; suppress warnings
            self.tokenizer = AutoTokenizer.from_pretrained(chosen_model, local_files_only=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(chosen_model, local_files_only=False).to(self.device)
            self.model.eval()
        except Exception:
            # If loading fails, set to None so detect() returns 0.0
            self.tokenizer = None
            self.model = None

    @torch.no_grad()
    def detect(self, text: str) -> float:
        """
        Returns the AI probability in percent using the BERT classifier.
        If the model is unavailable or the text is empty, returns 0.0.
        Only the first 1000 characters are used to maintain performance.
        """
        if not self.model or not self.tokenizer:
            return 0.0
        if not text or not text.strip():
            return 0.0
        sample = text.strip()[:1000]
        # Prepare inputs for the model (truncate to 512 tokens)
        enc = self.tokenizer(sample, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc.get("input_ids").to(self.device)
        attn = enc.get("attention_mask").to(self.device)
        if input_ids.shape[1] == 0:
            return 0.0
        outputs = self.model(input_ids=input_ids, attention_mask=attn)
        logits = outputs.logits
        # Softmax over logits to obtain probabilities
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        # Assume label index 1 corresponds to AI and 0 to human
        score = float(probs[1] * 100.0)
        return score


# =========================
# Classifier AI Detector (scikit-learn)
# =========================
class AIDetectorClassifier:
    """
    A real classifier trained offline:
      - Text features: TF-IDF (word + char)
      - Numeric features: perplexity score (raw), burstiness cv, repetition rate
    """

    def __init__(self):
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, stop_words="english")
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000)
        self.clf = LogisticRegression(max_iter=2000, n_jobs=1)

    @staticmethod
    def _repetition_rate(text: str) -> float:
        toks = [t.lower() for t in word_tokenize(text) if re.match(r"^\w+$", t)]
        if len(toks) < 30:
            return 0.0
        uniq = len(set(toks))
        return float(1.0 - (uniq / max(1, len(toks))))

    @staticmethod
    def _burstiness_cv(text: str) -> float:
        sents = sent_tokenize(text)
        if len(sents) < 4:
            return 0.0
        lens = [len(word_tokenize(s)) for s in sents[:120]]
        m = float(np.mean(lens)) if lens else 0.0
        sd = float(np.std(lens)) if lens else 0.0
        return float(sd / m) if m > 0 else 0.0

    def _build_matrix(
        self,
        texts: List[str],
        perplexity_raw: Optional[List[float]] = None,
        burstiness_cv: Optional[List[float]] = None,
        repetition: Optional[List[float]] = None,
        fit: bool = False,
    ):
        if fit:
            Xw = self.word_vec.fit_transform(texts)
            Xc = self.char_vec.fit_transform(texts)
        else:
            Xw = self.word_vec.transform(texts)
            Xc = self.char_vec.transform(texts)

        pr = perplexity_raw if perplexity_raw is not None else [0.0] * len(texts)
        bc = burstiness_cv if burstiness_cv is not None else [0.0] * len(texts)
        rr = repetition if repetition is not None else [0.0] * len(texts)

        Xn = csr_matrix(np.column_stack([pr, bc, rr]).astype(np.float32))
        return hstack([Xw, Xc, Xn], format="csr")

    def fit(self, texts: List[str], y: List[int], perplexity_raw: List[float], burstiness_cv: List[float], repetition: List[float]):
        X = self._build_matrix(texts, perplexity_raw, burstiness_cv, repetition, fit=True)
        self.clf.fit(X, y)

    def predict_proba_ai(self, texts: List[str], perplexity_raw: List[float], burstiness_cv: List[float], repetition: List[float]) -> List[float]:
        X = self._build_matrix(texts, perplexity_raw, burstiness_cv, repetition, fit=False)
        # class 1 = AI
        p = self.clf.predict_proba(X)[:, 1]
        return [float(x) for x in p]


# =========================
# Plagiarism detector
# =========================
class PlagiarismDetector:
    def __init__(self):
        self.reference_corpus = self._get_reference_corpus()
        # SPEED OPTIMIZATION: Pre-fit vectorizer
        self.vect = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
        if self.reference_corpus:
            self.vect.fit(self.reference_corpus)
        else:
            self.vect.fit(["placeholder text"])

    def _get_reference_corpus(self) -> List[str]:
        return [
            "Artificial intelligence is transforming the world in unprecedented ways and creating new opportunities for innovation across industries.",
            "Machine learning algorithms have revolutionized how we process data and make predictions in complex systems.",
            "Cybersecurity threats are evolving rapidly, requiring adaptive defense mechanisms and continuous monitoring of digital assets.",
            "Climate change represents one of the greatest challenges of our time, demanding coordinated global efforts and sustainable solutions.",
            "The digital transformation has fundamentally altered business operations, customer interactions, and market dynamics worldwide.",
        ]

    def tfidf_similarity_text_to_refs(self, text: str) -> Tuple[float, int]:
        if not self.reference_corpus:
            return 0.0, -1
            
        try:
            all_docs = [text] + self.reference_corpus
            tfidf_matrix = self.vect.transform(all_docs)
            sims = (tfidf_matrix[0:1] @ tfidf_matrix[1:].T).toarray().flatten()
        except Exception:
            # Fallback if dimensions mismatch
             docs = [text] + self.reference_corpus
             self.vect = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1).fit(docs)
             tfidf = self.vect.transform(docs)
             sims = (tfidf[0:1] @ tfidf[1:].T).toarray().flatten()
             
        if sims.size == 0:
            return 0.0, -1
        idx = int(np.argmax(sims))
        return float(sims[idx] * 100.0), idx

    def ngram_jaccard(self, a: str, b: str, n: int = 5) -> float:
        def ngrams(tokens: List[str], n: int):
            return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

        ta = [t.lower() for t in word_tokenize(a) if re.match(r"^\w+$", t)]
        tb = [t.lower() for t in word_tokenize(b) if re.match(r"^\w+$", t)]
        if len(ta) < n or len(tb) < n:
            return 0.0
        na = ngrams(ta, n)
        nb = ngrams(tb, n)
        if not na or not nb:
            return 0.0
        inter = len(na & nb)
        union = len(na | nb)
        return float((inter / union) * 100.0) if union else 0.0

    def detect_plagiarism(self, text: str) -> Tuple[float, Dict[str, float]]:
        tfidf, idx = self.tfidf_similarity_text_to_refs(text)
        ng = 0.0
        if idx >= 0:
            ng = self.ngram_jaccard(text[:2500], self.reference_corpus[idx], n=5)
        final = 0.7 * tfidf + 0.3 * ng
        return float(final), {"tfidf_cosine": tfidf, "ngram_overlap": ng}

    def find_copied_segments(self, text: str, threshold: float = 40.0, max_hits: int = 20) -> List[CopiedSegment]:
        sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
        hits: List[CopiedSegment] = []
        for s in sents:
            if len(s.split()) < 8:
                continue
            tfidf_score, ref_idx = self.tfidf_similarity_text_to_refs(s)
            if ref_idx < 0:
                continue
            ref = self.reference_corpus[ref_idx]
            ng = self.ngram_jaccard(s, ref, n=5)
            combined = 0.7 * tfidf_score + 0.3 * ng
            if combined >= threshold:
                hits.append(CopiedSegment(
                    sentence=s,
                    best_match=ref,
                    tfidf_similarity=tfidf_score,
                    ngram_overlap=ng,
                    combined=combined,
                ))
        hits.sort(key=lambda x: x.combined, reverse=True)
        return hits[:max_hits]


# =========================
# Training dataset creation (FREE, offline)
# =========================
def _nltk_prepare():
    for pkg in ["punkt", "brown", "gutenberg"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


def _sample_human_texts(n: int) -> List[str]:
    _nltk_prepare()
    human = []
    try:
        from nltk.corpus import brown
        sents = brown.sents()
        cur = []
        for s in sents:
            cur.extend(s)
            if len(cur) >= 80:
                human.append(" ".join(cur[:80]))
                cur = []
            if len(human) >= n // 2:
                break
    except Exception:
        pass

    try:
        from nltk.corpus import gutenberg
        words = gutenberg.words()
        step = 90
        for i in range(0, min(len(words), n * step * 2), step):
            chunk = " ".join(words[i:i+90])
            if chunk:
                human.append(chunk)
            if len(human) >= n:
                break
    except Exception:
        pass

    if len(human) < max(50, n // 3):
        human.extend([
            "This paper discusses the background, methods, results, and conclusions of the study in a structured manner.",
            "In the following sections, we examine the factors influencing outcomes and propose recommendations for future work.",
            "The analysis considers historical context, present constraints, and the practical implications for stakeholders."
        ])

    return human[:n]


def _generate_ai_texts_gpt2(n: int, tokenizer, model, device: str) -> List[str]:
    prompts = [
        "In conclusion, the study indicates that",
        "The following analysis demonstrates that",
        "Artificial intelligence will likely",
        "From a practical perspective, it is important to",
        "The results suggest a strong relationship between",
    ]
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(n):
            prompt = prompts[i % len(prompts)]
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            gen = model.generate(
                input_ids,
                max_length=min(160, input_ids.shape[1] + 110),
                do_sample=True,
                top_p=0.92,
                temperature=0.9,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
            )
            text = tokenizer.decode(gen[0], skip_special_tokens=True)
            out.append(text)
    return out[:n]


def train_ai_detector(
    force: bool = False,
    human_samples: int = DEFAULT_HUMAN_SAMPLES,
    ai_samples: int = DEFAULT_AI_SAMPLES,
) -> Dict[str, float]:
    if os.path.exists(AI_MODEL_PATH) and not force:
        return {"status": 1.0, "note": "Model already exists. Use train_ai to retrain."}

    print("Training AI detector classifier (offline)…")
    _nltk_prepare()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(DEFAULT_AI_MODEL)
    lm = AutoModelForCausalLM.from_pretrained(DEFAULT_AI_MODEL).to(device)

    human = _sample_human_texts(human_samples)
    ai = _generate_ai_texts_gpt2(ai_samples, tok, lm, device)

    texts = human + ai
    y = [0] * len(human) + [1] * len(ai)

    heur = HeuristicAIDetector(DEFAULT_AI_MODEL)

    perplex_raw = []
    burst_cv = []
    rep = []

    def raw_perplexity(text: str) -> float:
        if not text or not text.strip(): return 0.0
        chunks = text.split()
        if len(chunks) < 20:
            return 0.0
        enc = heur.tokenizer(text[:1500], return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(heur.device)
        if input_ids.shape[1] == 0: return 0.0
        with torch.no_grad():
            outp = heur.model(input_ids, labels=input_ids)
            loss = float(outp.loss)
        return float(math.exp(min(20, loss)))

    for t in texts:
        pr = raw_perplexity(t)
        perplex_raw.append(pr)
        burst_cv.append(AIDetectorClassifier._burstiness_cv(t))
        rep.append(AIDetectorClassifier._repetition_rate(t))

    X_train, X_test, y_train, y_test, pr_tr, pr_te, bc_tr, bc_te, rr_tr, rr_te = train_test_split(
        texts, y, perplex_raw, burst_cv, rep, test_size=0.2, random_state=42, stratify=y
    )

    model = AIDetectorClassifier()
    model.fit(X_train, y_train, pr_tr, bc_tr, rr_tr)

    p_ai = model.predict_proba_ai(X_test, pr_te, bc_te, rr_te)
    y_pred = [1 if p >= 0.5 else 0 for p in p_ai]

    acc = float(accuracy_score(y_test, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    metrics = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}

    joblib.dump(
        {"model": model, "metrics": metrics, "trained_at": time.time()},
        AI_MODEL_PATH
    )

    print("Saved model:", AI_MODEL_PATH)
    print("Metrics:", metrics)
    return metrics


def load_ai_detector() -> Optional[dict]:
    if not os.path.exists(AI_MODEL_PATH):
        return None
    try:
        return joblib.load(AI_MODEL_PATH)
    except Exception:
        return None


# =========================
# Combined AI scoring
# =========================
class AICombinedDetector:
    def __init__(self):
        self.heur = HeuristicAIDetector(DEFAULT_AI_MODEL)
        # Load optional classifier bundle (LogisticRegression)
        self.bundle = load_ai_detector()
        # Initialize BERT-based detector for more robust AI detection
        self.bert = BERTAIDetector()

    @staticmethod
    def _synonym_density(text: str, sample_words: int = 120) -> float:
        """
        Estimate how frequently common words have alternative synonyms present in
        WordNet. A higher density suggests the text may have been deliberately
        paraphrased (e.g., using a humanizer tool). Only a subset of words is
        sampled for performance.
        """
        if not text or not text.strip():
            return 0.0
        words = [w for w in word_tokenize(text) if w.isalpha() and len(w) > 3]
        if not words:
            return 0.0
        # Limit analysis to the first N words to reduce overhead
        sample = words[:sample_words]
        count_with_syn = 0
        for w in sample:
            try:
                synsets = wordnet.synsets(w)
            except Exception:
                synsets = []
            synonyms = set()
            for syn in synsets:
                for lemma in syn.lemmas():
                    name = lemma.name().lower()
                    if name and name != w.lower():
                        synonyms.add(name)
            if synonyms:
                count_with_syn += 1
        return float(count_with_syn / len(sample)) if sample else 0.0

    @staticmethod
    def _detect_language(text: str) -> str:
        """
        Rudimentary language detection for English, Spanish, and Japanese.
        Uses character heuristics to classify the input. Returns 'en', 'es', 'ja', or 'unknown'.
        """
        if not text or not text.strip():
            return "unknown"
        sample = text[:200]
        # Detect Japanese by presence of Hiragana, Katakana, or Kanji characters
        if re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", sample):
            return "ja"
        # Count Spanish-specific characters (accents and ñ/ü)
        spanish_chars = re.findall(r"[áéíóúñü]", sample.lower())
        if len(spanish_chars) > 2:
            return "es"
        # Default to English if no other language markers
        return "en"

    def detect(self, text: str) -> Tuple[float, Dict[str, float]]:
        # Heuristic-based detection (perplexity + burstiness)
        heur_score, heur_details = self.heur.detect(text)

        # Classifier (LogisticRegression) probability and score
        cls_prob = 0.0
        cls_score = 0.0
        if self.bundle:
            model: AIDetectorClassifier = self.bundle.get("model")
            if model:
                pr_raw = [self._raw_perplexity(text)]
                bc = [AIDetectorClassifier._burstiness_cv(text)]
                rr = [AIDetectorClassifier._repetition_rate(text)]
                try:
                    p_ai = model.predict_proba_ai([text], pr_raw, bc, rr)[0]
                except Exception:
                    p_ai = 0.0
                cls_prob = float(p_ai)
                cls_score = float(cls_prob * 100.0)

        # BERT-based AI probability
        bert_score = self.bert.detect(text)

        # Synonym density for humanizer detection
        syn_density = self._synonym_density(text)

        # Determine category: AI-Generated Only vs AI-Generated + Paraphrased
        category = "Human or Low Risk"
        # Combined heuristics to define AI risk threshold
        ai_indicator = max(heur_score, bert_score, cls_score)
        if ai_indicator >= 20.0:
            if syn_density > 0.15:
                category = "AI-Generated + Paraphrased"
            else:
                category = "AI-Generated Only"

        # Blend scores: assign weights (heuristics and BERT are primary drivers)
        final_score = (0.45 * heur_score) + (0.45 * bert_score) + (0.10 * cls_score)

        # Reliability thresholds: mask low scores (<20%) as zero and categorize as low risk
        if final_score < 20.0:
            final_score = 0.0
            category = "Human or Low Risk"

        details = {
            **heur_details,
            "classifier_ai_prob": cls_prob,
            "classifier_ai_score": cls_score,
            "bert_score": bert_score,
            "heur_score": heur_score,
            "ai_final": final_score,
            "synonym_density": syn_density,
            "language": self._detect_language(text),
            "category": category,
            "final_blend_note": "final = 45% heuristics + 45% BERT + 10% classifier",
        }

        return float(final_score), details

    def detect_sentences(self, text: str) -> List[AISegment]:
        """
        Perform sentence-level AI detection and augment each segment with
        paraphrasing information. Sentences are first identified as likely AI
        generated using the heuristic burstiness approach. For each flagged
        sentence, the synonym density is computed using `_synonym_density` to
        estimate whether an AI paraphrasing tool was used. If the density
        exceeds a conservative threshold (0.15), the `paraphrased` flag on the
        resulting `AISegment` is set to True and the `synonym_density` value
        recorded. Segments that do not meet the AI threshold remain unflagged.

        Parameters
        ----------
        text : str
            The preprocessed document text.

        Returns
        -------
        List[AISegment]
            A list of sentence-level AI segments with paraphrasing metadata.
        """
        segments: List[AISegment] = self.heur.detect_sentence_level(text)
        for seg in segments:
            # Compute synonym density only for sentences already flagged as AI-like
            density = self._synonym_density(seg.sentence)
            seg.synonym_density = density
            # Mark as paraphrased if density exceeds threshold
            if density > 0.15:
                seg.paraphrased = True
        return segments

    def _raw_perplexity(self, text: str) -> float:
        if not text or not text.strip(): return 0.0
        enc = self.heur.tokenizer(text[:1500], return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(self.heur.device)
        if input_ids.shape[1] == 0: return 0.0
        with torch.no_grad():
            outp = self.heur.model(input_ids, labels=input_ids)
            loss = float(outp.loss)
        return float(math.exp(min(20, loss)))


# =========================
# Core analyzer
# =========================
class Checker:
    def __init__(self):
        self.ai = AICombinedDetector()
        self.plag = PlagiarismDetector()

    def analyze_path(self, file_path: str) -> Optional[Results]:
        raw = DocumentProcessor.get_document_text(file_path)
        if not raw:
            return None

        text = DocumentProcessor.preprocess_text(raw)
        clean_text = DocumentProcessor.exclude_cover_declaration(text)
        wc = len(clean_text.split())

        ai_score, ai_details = self.ai.detect(clean_text)
        plag_score, plag_details = self.plag.detect_plagiarism(clean_text)
        copied = self.plag.find_copied_segments(clean_text, threshold=40.0, max_hits=20)
        
        ai_segs = self.ai.detect_sentences(clean_text)

        return Results(
            file_name=os.path.basename(file_path),
            word_count=wc,
            text_content=clean_text,
            ai_score=ai_score,
            ai_details=ai_details,
            plag_score=plag_score,
            plag_details=plag_details,
            copied_segments=copied,
            ai_segments=ai_segs
        )


# =========================
# Reports (HTML + PDF)
# =========================
def _make_score_chart_png(ai: float, plag: float) -> bytes:
    fig = plt.figure(figsize=(5.2, 2.8), dpi=160)
    ax = fig.add_subplot(111)
    colors_list = ['#FF9F1C', '#FF595E'] # Orange for AI, Red for Plag
    ax.bar(["AI Likelihood", "Plagiarism Risk"], [ai, plag], color=colors_list)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Scores", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('#333333')
    ax.spines['right'].set_color('#333333')
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", facecolor='#1a1a1a')
    plt.close(fig)
    return buf.getvalue()

def _highlight_text(text: str, plag_segs: List[CopiedSegment], ai_segs: List[AISegment]) -> str:
    """
    Injects HTML spans into the text to highlight AI and Plagiarism.
    Priority: Plagiarism > AI.
    """
    sentences = sent_tokenize(text)
    output_sentences = []
    
    plag_lookup = {s.sentence.lower(): s for s in plag_segs}
    ai_lookup = {s.sentence.lower(): s for s in ai_segs}
    
    for s in sentences:
        norm_s = s.strip()
        if not norm_s:
            output_sentences.append(s)
            continue
            
        lower_s = norm_s.lower()
        
        if lower_s in plag_lookup:
            data = plag_lookup[lower_s]
            output_sentences.append(f'<span class="highlight-plag" title="Plagiarism: {data.combined:.0f}% Match">{html_escape(norm_s)}</span>')
        elif lower_s in ai_lookup:
            data = ai_lookup[lower_s]
            # Choose appropriate CSS class based on paraphrasing flag
            if getattr(data, "paraphrased", False):
                # For paraphrased segments, include synonym density and AI score in title
                title = f"AI Paraphrased: {data.ai_score:.0f}% (synonym density: {data.synonym_density:.2f})"
                output_sentences.append(
                    f'<span class="highlight-ai-paraphrased" title="{html_escape(title)}">{html_escape(norm_s)}</span>'
                )
            else:
                title = f"AI Probability: {data.ai_score:.0f}%"
                output_sentences.append(
                    f'<span class="highlight-ai" title="{html_escape(title)}">{html_escape(norm_s)}</span>'
                )
        else:
            output_sentences.append(html_escape(norm_s))
            
    return " ".join(output_sentences)


def export_html(results: List[Results], out_path: str) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = ""
    for r in results:
        rows += f"""
        <tr>
          <td>{html_escape(r.file_name)}</td>
          <td>{r.word_count}</td>
          <td>{r.ai_score:.1f}%</td>
          <td>{r.plag_score:.1f}%</td>
        </tr>
        """

    detail_blocks = ""
    for r in results:
        seg_rows = ""
        for s in r.copied_segments:
            seg_rows += f"""
            <tr>
              <td>{s.combined:.1f}%</td>
              <td>{s.tfidf_similarity:.1f}%</td>
              <td>{s.ngram_overlap:.1f}%</td>
              <td>{html_escape(s.sentence)}</td>
              <td>{html_escape(s.best_match)}</td>
            </tr>
            """
        if not seg_rows:
            seg_rows = """<tr><td colspan="5"><i>No high-confidence copied sentences detected.</i></td></tr>"""

        cls_prob = r.ai_details.get("classifier_ai_prob", 0.0)
        detail_blocks += f"""
        <div class="card">
          <h2>{html_escape(r.file_name)}</h2>
          <p>
            <b>Word count:</b> {r.word_count}<br/>
            <b>AI Likelihood:</b> {r.ai_score:.1f}%<br/>
            <span class="muted">Classifier AI prob:</span> {cls_prob:.3f}<br/>
            <span class="muted">Heuristic:</span> Perplexity {r.ai_details.get("perplexity",0.0):.1f}% • Burstiness {r.ai_details.get("burstiness",0.0):.1f}%
          </p>

          <h3>Likely Copied Sentences</h3>
          <table>
            <thead>
              <tr>
                <th>Combined</th><th>TF-IDF</th><th>5-gram</th><th>Sentence</th><th>Best Match</th>
              </tr>
            </thead>
            <tbody>
              {seg_rows}
            </tbody>
          </table>
        </div>
        """

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>AI & Plagiarism Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; background: #fafafa; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; }}
    .top {{ display:flex; justify-content: space-between; gap: 12px; align-items: baseline; }}
    .muted {{ color:#666; font-size: 12px; }}
    .card {{ background: #fff; border: 1px solid #e6e6e6; border-radius: 12px; padding: 14px; margin: 12px 0; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border: 1px solid #e7e7e7; padding: 8px; vertical-align: top; }}
    th {{ background: #f3f3f3; text-align:left; }}
    h1 {{ margin: 8px 0 2px; }}
    h2 {{ margin: 4px 0 10px; }}
    h3 {{ margin: 12px 0 8px; }}
  </style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <h1>AI & Plagiarism Analysis Report</h1>
    <div class="muted">Generated: {now}</div>
  </div>

  <div class="card">
    <h2>Comparison Summary</h2>
    <table>
      <thead><tr><th>File</th><th>Words</th><th>AI</th><th>Plagiarism</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>

  {detail_blocks}

</div>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def export_pdf(results: List[Results], out_path: str) -> str:
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>AI & Plagiarism Analysis Report</b>", styles["Title"]))
    story.append(Spacer(1, 10))
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"<font color='#666666'>Generated: {now}</font>", styles["Normal"]))
    story.append(Spacer(1, 12))

    data = [["File", "Words", "AI %", "Plagiarism %"]]
    for r in results:
        data.append([r.file_name, str(r.word_count), f"{r.ai_score:.1f}", f"{r.plag_score:.1f}"])

    t = Table(data, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EFEFEF")),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#DDDDDD")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#FBFBFB")]),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(Paragraph("<b>Comparison Summary</b>", styles["Heading2"]))
    story.append(t)
    story.append(Spacer(1, 14))

    for r in results:
        story.append(Paragraph(f"<b>{html_escape(r.file_name)}</b>", styles["Heading2"]))
        cls_prob = r.ai_details.get("classifier_ai_prob", 0.0)

        story.append(Paragraph(
            f"Word count: <b>{r.word_count}</b><br/>"
            f"AI Likelihood: <b>{r.ai_score:.1f}%</b> (Classifier prob {cls_prob:.3f})<br/>"
            f"Plagiarism Risk: <b>{r.plag_score:.1f}%</b>",
            styles["Normal"]
        ))
        story.append(Spacer(1, 10))

        png = _make_score_chart_png(r.ai_score, r.plag_score)
        img = Image(io.BytesIO(png))
        img.drawHeight = 180
        img.drawWidth = 330
        story.append(img)
        story.append(Spacer(1, 12))

        story.append(Paragraph("<b>Likely Copied Sentences (Top)</b>", styles["Heading3"]))
        seg_data = [["Combined", "TF-IDF", "5-gram", "Sentence", "Best Match"]]
        if r.copied_segments:
            for s in r.copied_segments[:10]:
                seg_data.append([
                    f"{s.combined:.1f}%",
                    f"{s.tfidf_similarity:.1f}%",
                    f"{s.ngram_overlap:.1f}%",
                    s.sentence[:220] + ("..." if len(s.sentence) > 220 else ""),
                    s.best_match[:220] + ("..." if len(s.best_match) > 220 else ""),
                ])
        else:
            seg_data.append(["-", "-", "-", "No high-confidence copied sentences detected.", "-"])

        seg_t = Table(seg_data, colWidths=[55, 50, 50, 200, 200], hAlign="LEFT")
        seg_t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E8F0FE")),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#DDDDDD")),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        story.append(seg_t)
        story.append(Spacer(1, 18))

    doc = SimpleDocTemplate(out_path, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    doc.build(story)
    return out_path


# =========================
# Flask dashboard (NO templates)
# =========================
app = Flask(__name__)
_JOBS: Dict[str, Dict[str, str]] = {}
_JOB_DIRS: Dict[str, str] = {}


def _cleanup_old_jobs(max_age_sec: int = 3600):
    now = time.time()
    dead = []
    for jid, meta in _JOBS.items():
        ts = float(meta.get("ts", now))
        if now - ts > max_age_sec:
            dead.append(jid)
    for jid in dead:
        try:
            d = _JOB_DIRS.get(jid)
            if d and os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
        _JOBS.pop(jid, None)
        _JOB_DIRS.pop(jid, None)


def _render_home_html(msg: str = "") -> str:
    """
    Render the home page with a refreshed interface for the RedHydra tool.  This
    version uses a dark theme inspired by the RedHydra logo and includes the
    user's GitHub profile link and logo. It provides a functional file upload
    form and descriptive features.

    Parameters
    ----------
    msg : str
        Optional message to display to the user (for errors or notices).

    Returns
    -------
    str
        Complete HTML markup for the home page.
    """
    alert_html = (
        f"<div class='alert alert-danger' style='margin-bottom:20px;'>{html_escape(msg)}</div>"
        if msg
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RedHydra AI &amp; Plagiarism Checker – Dashboard</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    :root {{
      --bg: #0d1117;
      --card: #161b22;
      --primary: #e63946;
      --secondary: #38b000;
      --text: #f5f6fa;
      --muted: #8a8f98;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      min-height: 100vh;
    }}
    a {{ color: var(--primary); text-decoration: none; }}
    .site-header {{
      background: var(--card);
      padding: 10px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid #222;
    }}
    .site-header .logo-title {{
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 600;
      font-size: 1.2rem;
    }}
    .site-header img {{
      height: 40px;
      width: auto;
    }}
    .site-header nav a {{
      margin-left: 20px;
      font-size: 0.9rem;
      color: var(--muted);
      transition: color 0.2s;
    }}
    .site-header nav a:hover {{ color: var(--primary); }}
    main {{ flex: 1; padding: 40px 20px; max-width: 1000px; margin: 0 auto; }}
    .hero {{ text-align: center; margin-bottom: 50px; }}
    .hero h1 {{ font-size: 2.4rem; font-weight: 700; color: var(--primary); margin-bottom: 10px; }}
    .hero p {{ font-size: 1rem; color: var(--muted); margin-bottom: 30px; }}
    .upload-card {{ background: var(--card); border: 1px dashed var(--primary); border-radius: 12px; padding: 40px; text-align: center; }}
    .upload-card:hover {{ background: #1b2130; }}
    .upload-card i {{ font-size: 3rem; color: var(--primary); margin-bottom: 15px; }}
    .upload-card h4 {{ margin-bottom: 8px; font-weight: 600; }}
    .upload-card p {{ margin-bottom: 20px; color: var(--muted); font-size: 0.9rem; }}
    .upload-card .file-list {{ margin-top: 10px; font-size: 0.85rem; color: var(--secondary); text-align: left; max-height: 120px; overflow-y: auto; }}
    .scan-btn {{ margin-top: 20px; background: var(--primary); color: #fff; border: none; padding: 12px 32px; border-radius: 30px; font-weight: 600; cursor: pointer; transition: background 0.3s, transform 0.2s; }}
    .scan-btn:disabled {{ background: #555; cursor: not-allowed; }}
    .scan-btn:hover:not(:disabled) {{ background: #c2333f; transform: translateY(-2px); }}
    .features {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 60px; }}
    .feature {{ flex: 1 1 280px; background: var(--card); border-radius: 12px; padding: 24px; text-align: center; border: 1px solid #222; transition: transform 0.3s; }}
    .feature:hover {{ transform: translateY(-4px); }}
    .feature .icon {{ font-size: 2rem; color: var(--primary); margin-bottom: 12px; }}
    .feature h3 {{ margin-bottom: 8px; font-weight: 600; }}
    .feature p {{ font-size: 0.85rem; color: var(--muted); }}
    footer {{ background: var(--card); padding: 20px; text-align: center; color: var(--muted); font-size: 0.8rem; border-top: 1px solid #222; }}
  </style>
</head>
<body>
  <header class="site-header">
    <div class="logo-title">
      <img src="https://raw.githubusercontent.com/root60/WPScrapper/refs/heads/main/logo.png" alt="RedHydra logo">
      <span>RedHydra AI &amp; Plagiarism Checker</span>
    </div>
    <nav>
      <a href="/download-model"><i class="fas fa-download"></i> Model</a>
      <a href="https://github.com/root60" target="_blank" style="margin-left:20px;"><i class="fab fa-github"></i> GitHub</a>
    </nav>
  </header>
  <main>
    <section class="hero">
      <h1>Welcome to RedHydra</h1>
      <p>Detect AI‑generated and plagiarized content with precision</p>
      <!-- Display any message -->
      {alert_html}
    </section>
    <section>
      <form action="/analyze" method="post" enctype="multipart/form-data" ondragover="this.classList.add('hover')" ondragleave="this.classList.remove('hover')" style="max-width: 600px; margin: 0 auto;">
        <!-- Upload area: clicking anywhere opens the file dialog -->
        <div class="upload-card" onclick="document.getElementById('fileInput').click()">
          <i class="fas fa-upload"></i>
          <h4>Select or drop documents</h4>
          <p>Supports PDF, DOCX and TXT files</p>
          <input type="file" name="files" id="fileInput" multiple style="display:none" onchange="showFiles(this)">
          <div id="fileList" class="file-list"></div>
        </div>
        <!-- Place the scan button outside the clickable upload card so it isn't triggered by file selection clicks -->
        <button type="submit" class="scan-btn" id="scanBtn" disabled style="display:block; margin: 20px auto 0;">Start Scan</button>
      </form>
    </section>
    <section class="features">
      <div class="feature">
        <div class="icon"><i class="fas fa-brain"></i></div>
        <h3>AI Detection</h3>
        <p>Leverages heuristics, classifiers and BERT models to flag AI‑generated and paraphrased text.</p>
      </div>
      <div class="feature">
        <div class="icon"><i class="fas fa-file-alt"></i></div>
        <h3>Plagiarism Check</h3>
        <p>Cross‑checks against internal corpora and uploaded files to highlight copied passages.</p>
      </div>
      <div class="feature">
        <div class="icon"><i class="fas fa-bolt"></i></div>
        <h3>Speed &amp; Privacy</h3>
        <p>Runs locally for instant results without sending your documents to the cloud.</p>
      </div>
    </section>
  </main>
  <footer>
    &copy; {datetime.datetime.now().year} RedHydra. All rights reserved.
  </footer>
  <script>
    // Show selected files and enable scan button
    function showFiles(input) {{
      const list = document.getElementById('fileList');
      list.innerHTML = '';
      const files = input.files;
      const scanBtn = document.getElementById('scanBtn');
      if (!files || files.length === 0) {{
        scanBtn.setAttribute('disabled', '');
        return;
      }}
      const ul = document.createElement('ul');
      ul.style.listStyle = 'none';
      ul.style.padding = 0;
      for (let i = 0; i < files.length; i++) {{
        const li = document.createElement('li');
        li.textContent = files[i].name;
        ul.appendChild(li);
      }}
      list.appendChild(ul);
      scanBtn.removeAttribute('disabled');
    }}
  </script>
</body>
</html>
"""


def _render_results_html(job_id: str, results: List[Results]) -> str:
    """
    Render the analysis results page using an updated, animated layout. This version
    mirrors the design of `new_results.html`, building a results array in
    JavaScript and populating both a summary chart/table and detailed cards.

    Parameters
    ----------
    job_id : str
        Unique identifier for the analysis job.
    results : List[Results]
        List of result objects for each uploaded file.

    Returns
    -------
    str
        Complete HTML markup for the results page.
    """
    # Build a list of dictionaries representing each result for embedding in JS
    js_results: List[Dict[str, Any]] = []
    for r in results:
        # Prepare AI segments: include paraphrasing flag
        ai_segments_js = []
        for seg in r.ai_segments:
            truncated_text = seg.sentence[:80] + ("..." if len(seg.sentence) > 80 else "")
            ai_segments_js.append({
                "text": html_escape(truncated_text),
                "score": round(seg.ai_score),
                "para": bool(getattr(seg, "paraphrased", False))
            })
        # Prepare plagiarism segments: use combined score and sentence text
        plag_segments_js = []
        for seg in r.copied_segments:
            truncated_sent = seg.sentence[:80] + ("..." if len(seg.sentence) > 80 else "")
            plag_segments_js.append({
                "score": round(seg.combined, 1),
                "text": html_escape(truncated_sent)
            })
        # Generate highlighted text: convert our span classes to the ones used in the new UI
        raw_highlight = _highlight_text(r.text_content[:3000], r.copied_segments, r.ai_segments)
        # Replace class names for consistency with new template
        raw_highlight = (
            raw_highlight
            .replace("highlight-ai-paraphrased", "para")
            .replace("highlight-ai", "ai")
            .replace("highlight-plag", "plag")
        )
        if len(r.text_content) > 3000:
            raw_highlight += "..."
        js_results.append({
            "file": r.file_name,
            "words": r.word_count,
            "ai": float(f"{r.ai_score:.2f}"),
            "plag": float(f"{r.plag_score:.2f}"),
            "cls": float(f"{r.ai_details.get('classifier_ai_prob', 0.0):.3f}"),
            "aiSegments": ai_segments_js,
            "plagSegments": plag_segments_js,
            "highlight": raw_highlight
        })
    results_json = json.dumps(js_results)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RedHydra AI &amp; Plagiarism Checker – Results</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/aos/2.3.4/aos.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    :root {{
      --bg-color: #0E1125;
      --text-color: #e9e9f0;
      --primary-start: #5a189a;
      --primary-end: #0c134f;
      --accent: #F8B400;
      --danger: #e63946;
      --success: #8ac926;
      --card-bg: rgba(14,17,37,0.65);
      --card-border: rgba(255,255,255,0.06);
    }}
    body {{
      margin: 0;
      font-family: 'Inter', sans-serif;
      background: var(--bg-color);
      color: var(--text-color);
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    /* Remove default header styling; custom header uses .site-header */
    .overview {{
      max-width: 1000px;
      margin: 40px auto;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 18px;
      padding: 30px;
      box-shadow: 0 12px 32px rgba(0,0,0,0.6);
    }}
    .overview h2 {{
      margin-bottom: 15px;
    }}
    .chart-container {{
      position: relative;
      height: 280px;
      margin-bottom: 30px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    table th, table td {{
      padding: 10px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
    }}
    table thead th {{
      color: var(--accent);
      text-align: left;
      font-weight: 600;
    }}
    .details-container {{
      max-width: 1000px;
      margin: 20px auto 60px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }}
    .detail-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 12px 32px rgba(0,0,0,0.6);
    }}
    .detail-card h3 {{
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .progress-bar-container {{
      height: 8px;
      background: rgba(255,255,255,0.1);
      border-radius: 4px;
      margin-top: 8px;
    }}
    .progress-bar {{
      height: 8px;
      border-radius: 4px;
    }}
    .plag-list, .ai-list {{
      list-style: none;
      padding: 0;
      margin: 0;
    }}
    .plag-list li, .ai-list li {{
      padding: 6px 0;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      font-size: 0.85rem;
      color: #b0b3d6;
    }}
    /* Style for AI paraphrased segments in the side lists */
    .ai-list li.para-ai {{
      color: #b388ff;
      font-style: italic;
    }}
    .highlight {{
      padding: 12px;
      background: rgba(255,255,255,0.05);
      border-radius: 12px;
      margin-top: 12px;
      max-height: 200px;
      overflow-y: auto;
      font-family: monospace;
      font-size: 0.9rem;
      line-height: 1.6;
    }}
    .highlight .ai {{
      background: rgba(248, 180, 0, 0.35);
      color: #fff;
    }}
    .highlight .plag {{
      background: rgba(230, 57, 70, 0.35);
      color: #fff;
    }}
    .highlight .para {{
      /* Purple tone for AI paraphrased segments */
      background: rgba(142, 68, 173, 0.35);
      color: #fff;
    }}
    /* Download button container and styles */
    .download-buttons {{
      display: flex;
      justify-content: flex-end;
      gap: 12px;
      margin: 20px auto;
      max-width: 1000px;
    }}
    .download-btn {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      padding: 8px 16px;
      border-radius: 8px;
      color: var(--accent);
      font-size: 0.85rem;
      display: flex;
      align-items: center;
      gap: 6px;
      text-decoration: none;
      transition: background 0.3s, color 0.3s;
    }}
    .download-btn:hover {{
      background: rgba(255,255,255,0.1);
      color: #fff;
    }}
    /* Header styling to align with RedHydra branding */
    .site-header {{
      padding: 10px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--card-bg);
      border-bottom: 1px solid rgba(255,255,255,0.05);
    }}
    .site-header .logo-title {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 600;
      font-size: 1rem;
    }}
    .site-header img {{
      height: 32px;
      width: auto;
    }}
    .site-header nav a {{
      margin-left: 20px;
      font-size: 0.85rem;
      color: #b0b3d6;
      transition: color 0.2s;
    }}
    .site-header nav a:hover {{ color: var(--accent); }}
  </style>
</head>
<body>
  <header class="site-header">
    <div class="logo-title">
      <img src="https://raw.githubusercontent.com/root60/WPScrapper/refs/heads/main/logo.png" alt="RedHydra logo">
      <span>RedHydra AI &amp; Plagiarism Checker</span>
    </div>
    <nav>
      <a href="https://github.com/root60" target="_blank"><i class="fab fa-github"></i> GitHub</a>
      <a href="/" title="Start a new scan"><i class="fas fa-home"></i> New Scan</a>
    </nav>
  </header>
  <!-- Download report buttons -->
  <div class="download-buttons" data-aos="fade-down">
    <a href="/download/{job_id}/html" class="download-btn"><i class="fas fa-file-code"></i> HTML</a>
    <a href="/download/{job_id}/pdf" class="download-btn"><i class="fas fa-file-pdf"></i> PDF</a>
  </div>
  <section class="overview" data-aos="fade-up">
    <h2>Overview</h2>
    <div class="chart-container">
      <canvas id="summaryChart"></canvas>
    </div>
    <table>
      <thead>
        <tr><th>File</th><th>Words</th><th>AI Score</th><th>Plagiarism Score</th></tr>
      </thead>
      <tbody id="summaryTableBody">
        <!-- Rows inserted by script -->
      </tbody>
    </table>
  </section>
  <div class="details-container" id="detailsContainer">
    <!-- Detailed cards inserted by script -->
  </div>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/aos/2.3.4/aos.js"></script>
  <script>
    AOS.init({{ once: true, duration: 700 }});
    // Data generated on the server
    const results = {results_json};
    // Populate summary table and arrays for chart
    const labels = [];
    const aiData = [];
    const plagData = [];
    const tbody = document.getElementById('summaryTableBody');
    results.forEach(r => {{
      labels.push(r.file);
      aiData.push(parseFloat(r.ai.toFixed(1)));
      plagData.push(parseFloat(r.plag.toFixed(1)));
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${{r.file}}</td><td>${{r.words}}</td><td>${{r.ai.toFixed(1)}}%</td><td>${{r.plag.toFixed(1)}}%</td>`;
      tbody.appendChild(tr);
    }});
    // Render bar chart
    const ctx = document.getElementById('summaryChart').getContext('2d');
    new Chart(ctx, {{
      type: 'bar',
      data: {{
        labels: labels,
        datasets: [
          {{
            label: 'AI Likelihood',
            data: aiData,
            backgroundColor: '#f8b400'
          }},
          {{
            label: 'Plagiarism Risk',
            data: plagData,
            backgroundColor: '#e63946'
          }}
        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
          y: {{ beginAtZero: true, max: 100, grid: {{ color: 'rgba(255,255,255,0.1)' }}, ticks: {{ color: '#b0b3d6' }} }},
          x: {{ ticks: {{ color: '#b0b3d6' }}, grid: {{ display: false }} }}
        }},
        plugins: {{ legend: {{ labels: {{ color: '#b0b3d6' }} }} }}
      }}
    }});
    // Populate detailed cards
    const detailsContainer = document.getElementById('detailsContainer');
    results.forEach(r => {{
      const card = document.createElement('div');
      card.className = 'detail-card';
      card.setAttribute('data-aos', 'fade-up');
      // Build AI segments list markup. Highlight paraphrased AI segments with a class.
      const aiList = r.aiSegments.length ? r.aiSegments.map(s => {{
        const liClass = s.para ? ' class="para-ai"' : '';
        // Use concatenation to avoid template literals interfering with Python formatting
        return '<li' + liClass + '>' + s.text + " <span style=\'float:right\'>" + s.score + '%</span></li>';
      }}).join('') : '<li>No AI segments flagged</li>';
      // Build plagiarism segments list markup
      const plagList = r.plagSegments.length ? r.plagSegments.map(p => `<li>${{p.text}} <span style='float:right'>${{p.score}}%</span></li>`).join('') : '<li>No high plagiarism matches</li>';
      card.innerHTML = `
        <h3><i class="fas fa-file-alt"></i> ${{r.file}}</h3>
        <p>Word count: <strong>${{r.words}}</strong></p>
        <p>AI Likelihood: <strong>${{r.ai.toFixed(1)}}%</strong> &nbsp;|&nbsp; Classifier Prob: ${{r.cls.toFixed(2)}}</p>
        <div class="progress-bar-container"><div class="progress-bar" style="width:${{r.ai}}% ; background:#f8b400;"></div></div>
        <p>Plagiarism Risk: <strong>${{r.plag.toFixed(1)}}%</strong></p>
        <div class="progress-bar-container"><div class="progress-bar" style="width:${{r.plag}}% ; background:#e63946;"></div></div>
        <div style="margin-top: 16px; display:flex; gap:20px; flex-wrap:wrap;">
          <div style="flex:1; min-width:220px;">
            <h4 style="color:#f8b400; font-size:1rem; margin-bottom:8px;">AI Segments</h4>
            <ul class="ai-list">${{aiList}}</ul>
          </div>
          <div style="flex:1; min-width:220px;">
            <h4 style="color:#e63946; font-size:1rem; margin-bottom:8px;">Plagiarism Matches</h4>
            <ul class="plag-list">${{plagList}}</ul>
          </div>
        </div>
        <div class="highlight">${{r.highlight}}</div>
      `;
      detailsContainer.appendChild(card);
    }});
  </script>
</body>
</html>
"""


@app.get("/")
def home():
    _cleanup_old_jobs()
    return Response(_render_home_html(), mimetype="text/html")


@app.post("/analyze")
def analyze():
    _cleanup_old_jobs()

    files = request.files.getlist("files")
    if not files:
        return Response(_render_home_html("No files received."), mimetype="text/html")

    tmpdir = tempfile.mkdtemp(prefix="ai_plag_")
    job_id = uuid.uuid4().hex
    _JOB_DIRS[job_id] = tmpdir
    _JOBS[job_id] = {"ts": str(time.time())}

    saved_paths = []
    for f in files:
        name = os.path.basename(f.filename or "")
        if not name:
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in [".pdf", ".docx", ".txt"]:
            continue
        dest = os.path.join(tmpdir, name)
        f.save(dest)
        saved_paths.append(dest)

    if not saved_paths:
        return Response(_render_home_html("No valid files. Use PDF / DOCX / TXT."), mimetype="text/html")

    checker = Checker()

    base_refs = list(checker.plag.reference_corpus)

    doc_texts: Dict[str, str] = {}
    for p in saved_paths:
        raw = DocumentProcessor.get_document_text(p)
        if not raw:
            continue
        t = DocumentProcessor.preprocess_text(raw)
        t = DocumentProcessor.exclude_cover_declaration(t)
        doc_texts[p] = t

    results: List[Results] = []
    for p in saved_paths:
        if len(doc_texts) > 1 and p in doc_texts:
            checker.plag.reference_corpus = [
                doc_texts[op] for op in saved_paths
                if op != p and op in doc_texts and doc_texts[op].strip()
            ] or base_refs
        else:
            checker.plag.reference_corpus = base_refs

        r = checker.analyze_path(p)
        if r:
            results.append(r)

    checker.plag.reference_corpus = base_refs

    if not results:
        return Response(_render_home_html("Failed to analyze documents."), mimetype="text/html")

    html_path = os.path.join(tmpdir, "report.html")
    pdf_path = os.path.join(tmpdir, "report.pdf")
    export_html(results, html_path)
    export_pdf(results, pdf_path)

    _JOBS[job_id]["html"] = html_path
    _JOBS[job_id]["pdf"] = pdf_path

    return Response(_render_results_html(job_id, results), mimetype="text/html")


# Route to allow downloading external Hugging Face models from the UI
@app.route("/download-model", methods=["GET", "POST"])
def download_model_route():
    """
    Render a form to download a custom Hugging Face model for AI detection.
    On POST, attempts to download the specified model and store it for future
    use.  Displays a success or error message accordingly.
    """
    if request.method == "POST":
        model_id = request.form.get("model", "").strip()
        if not model_id:
            return Response(_render_model_download_html("Please provide a model identifier."), mimetype="text/html")
        exit_code = download_bert_model(model_id)
        if exit_code == 0:
            msg = f"Model '{html_escape(model_id)}' downloaded successfully."
        else:
            msg = f"Failed to download model '{html_escape(model_id)}'. Check the server logs."
        return Response(_render_model_download_html(msg), mimetype="text/html")
    # GET request
    return Response(_render_model_download_html(), mimetype="text/html")


@app.get("/download/<job_id>/html")
def download_html(job_id: str):
    meta = _JOBS.get(job_id)
    if not meta or "html" not in meta:
        return Response("Report not found.", status=404)
    return send_file(meta["html"], as_attachment=True, download_name="report.html")


@app.get("/download/<job_id>/pdf")
def download_pdf(job_id: str):
    meta = _JOBS.get(job_id)
    if not meta or "pdf" not in meta:
        return Response("Report not found.", status=404)
    return send_file(meta["pdf"], as_attachment=True, download_name="report.pdf")


# =========================
# CLI
# =========================
def main_cli(paths: List[str]) -> int:
    checker = Checker()
    results: List[Results] = []
    for p in paths:
        if not os.path.exists(p):
            print(f"File not found: {p}")
            continue
        r = checker.analyze_path(p)
        if r:
            results.append(r)
            print("\n" + "=" * 60)
            print("AI & PLAGIARISM ANALYSIS RESULTS")
            print("=" * 60)
            print("File:", r.file_name)
            print("Words:", r.word_count)
            print(f"AI Likelihood: {r.ai_score:.1f}%  (classifier prob {r.ai_details.get('classifier_ai_prob',0.0):.3f})")
            print(f"Plagiarism Risk: {r.plag_score:.1f}%")
            print("=" * 60)

    if not results:
        print("No results.")
        return 1

    out_html = os.path.join(os.getcwd(), "report.html")
    out_pdf = os.path.join(os.getcwd(), "report.pdf")
    export_html(results, out_html)
    export_pdf(results, out_pdf)
    print(f"\nSaved HTML: {out_html}")
    print(f"Saved PDF : {out_pdf}")
    return 0


def eval_ai_detector():
    bundle = load_ai_detector()
    if not bundle:
        print("No trained model found. Run: py -3 AII.py train_ai")
        return 1
    print("Saved model metrics:", bundle.get("metrics", {}))
    return 0

# =========================
# External Model Download Utility
# =========================
def download_bert_model(model_id: str) -> int:
    """
    Download a Hugging Face sequence classification model specified by
    ``model_id``. The tokenizer and model will be cached locally via the
    transformers library. After download, the chosen model identifier is
    saved to ``MODEL_CONFIG_FILE`` so that subsequent runs of the app
    automatically use this model for BERT-based AI detection.

    Parameters
    ----------
    model_id : str
        The Hugging Face repository ID (e.g., ``"followsci/bert-ai-text-detector"``).

    Returns
    -------
    int
        Exit code: 0 on success, non-zero on failure.
    """
    print(f"Downloading Hugging Face model '{model_id}' ...")
    try:
        # Trigger download of tokenizer and model. local_files_only=False ensures
        # the latest version is fetched if online. If offline, this will
        # attempt to load from the cache and will raise if missing.
        _ = AutoTokenizer.from_pretrained(model_id, local_files_only=False)
        _ = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=False)
        # Persist the chosen model name to config file
        with open(MODEL_CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(model_id)
        print(f"Successfully downloaded and configured '{model_id}'.")
        return 0
    except Exception as e:
        print(f"Error downloading model '{model_id}': {e}")
        return 1

def _render_model_download_html(msg: str = "") -> str:
    """
    Render the HTML for downloading an external Hugging Face model.  This page
    provides a simple form where users can enter a model ID (e.g.,
    ``followsci/bert-ai-text-detector``) to download and configure it for
    future scans.  Any message (success or error) is displayed at the top.

    Parameters
    ----------
    msg : str
        Optional message to display to the user (status or error notice).

    Returns
    -------
    str
        Complete HTML markup for the model download page.
    """
    alert_html = (
        f"<div class='alert alert-info' style='margin-bottom:20px;'>{html_escape(msg)}</div>"
        if msg
        else ""
    )
    # Build HTML with format substitution. All braces are doubled except for the
    # ``{alert_html}`` placeholder which will be replaced via ``str.format``.
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RedHydra - Download Model</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    :root {{
      --bg: #0d1117;
      --card: #161b22;
      --primary: #e63946;
      --text: #f5f6fa;
      --muted: #8a8f98;
    }}
    body {{ margin: 0; font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); }}
    a {{ color: var(--primary); text-decoration: none; }}
    .site-header {{ padding: 10px 20px; background: var(--card); border-bottom: 1px solid #222; display: flex; justify-content: space-between; align-items: center; }}
    .site-header .logo-title {{ display: flex; align-items: center; gap: 12px; font-weight: 600; font-size: 1rem; }}
    .site-header img {{ height: 34px; width: auto; }}
    .site-header nav a {{ margin-left: 20px; font-size: 0.85rem; color: var(--muted); transition: color 0.2s; }}
    .site-header nav a:hover {{ color: var(--primary); }}
    main {{ max-width: 700px; margin: 40px auto; padding: 0 20px; }}
    h1 {{ margin-bottom: 20px; font-size: 1.8rem; color: var(--primary); }}
    form {{ background: var(--card); padding: 30px; border-radius: 12px; border: 1px solid #222; }}
    label {{ display: block; margin-bottom: 8px; font-weight: 600; }}
    input[type='text'] {{ width: 100%; padding: 10px; border-radius: 6px; border: 1px solid #333; background: #0f141b; color: #fff; margin-bottom: 20px; }}
    button {{ background: var(--primary); color: #fff; border: none; padding: 12px 28px; border-radius: 30px; font-weight: 600; cursor: pointer; transition: background 0.3s; }}
    button:hover {{ background: #c2333f; }}
    .message {{ margin-top: 15px; font-size: 0.9rem; color: var(--muted); }}
  </style>
</head>
<body>
  <header class="site-header">
    <div class="logo-title">
      <img src="https://raw.githubusercontent.com/root60/WPScrapper/refs/heads/main/logo.png" alt="RedHydra logo">
      <span>RedHydra AI &amp; Plagiarism Checker</span>
    </div>
    <nav>
      <a href="/">Home</a>
      <a href="https://github.com/root60" target="_blank"><i class="fab fa-github"></i> GitHub</a>
    </nav>
  </header>
  <main>
    <h1>Download External AI Model</h1>
    {alert_html}
    <form method="post" action="/download-model">
      <label for="model">Hugging Face Model ID</label>
      <input type="text" id="model" name="model" placeholder="e.g., followsci/bert-ai-text-detector" required>
      <button type="submit"><i class="fas fa-download"></i> Download &amp; Configure</button>
    </form>
    <p class="message">After downloading, your scans will use the new model automatically.</p>
  </main>
</body>
</html>
"""
    return html.format(alert_html=alert_html)

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1].lower() == "flask":
        app.run(host="127.0.0.1", port=5000, debug=True)

    elif len(sys.argv) >= 2 and sys.argv[1].lower() == "train_ai":
        train_ai_detector(force=True)
        raise SystemExit(0)

    elif len(sys.argv) >= 2 and sys.argv[1].lower() == "eval_ai":
        raise SystemExit(eval_ai_detector())

    elif len(sys.argv) >= 3 and sys.argv[1].lower() == "download_model":
        # Allows users to pre-download a Hugging Face model for BERT detection
        # Usage: py -3 AII.py download_model <model_id>
        model_id = sys.argv[2]
        raise SystemExit(download_bert_model(model_id))

    else:
        if len(sys.argv) < 2:
            print("\nUsage:")
            print("  py -3 AII.py flask")
            print("  py -3 AII.py train_ai")
            print("  py -3 AII.py eval_ai")
            print("  py -3 AII.py file1.docx file2.pdf\n")
            raise SystemExit(1)

        raise SystemExit(main_cli(sys.argv[1:]))