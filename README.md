
<!-- ===================== REDHYDRA README ===================== -->

<p align="center">
  <img src="https://raw.githubusercontent.com/root60/WPScrapper/refs/heads/main/logo.png" width="140"/>
</p>

<h1 align="center">RedHydra AI & Plagiarism Checker</h1>

<p align="center">
  <b>Offline • Explainable • Unlimited • Research‑Grade</b><br/>
  Advanced AI‑Writing, AI‑Paraphrase & Plagiarism Detection Engine
</p>

<p align="center">
  <!-- Animated / Dynamic SVG Badges -->
  <img src="https://img.shields.io/badge/STATUS-ACTIVE-brightgreen.svg?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/OFFLINE-READY-blue.svg?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/AI-DETECTION-red.svg?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/PARAPHRASE-DETECTION-purple.svg?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/LICENSE-OPEN--SOURCE-orange.svg?style=for-the-badge"/>
</p>

---

## 🔥 Visual Overview

<p align="center">
  <img src="dashboard.png" width="95%"/>
</p>

<p align="center">
  <img src="download external model.png" width="95%"/>
</p>

<p align="center">
  <img src="result.png" width="95%"/>
</p>

---

## 🎯 Purpose

**RedHydra** is built to solve a modern problem:

> _How do we reliably detect AI‑generated and AI‑paraphrased writing **without cloud services, black boxes, or usage limits**?_

RedHydra answers this by combining:
- Transparent heuristics
- Offline ML classifiers
- Transformer‑based AI detectors
- Visual, explainable reporting

---

## ⚙️ System Architecture

```mermaid
flowchart TD
    A[User Uploads Documents] --> B[Preprocessing Engine]
    B --> C[Plagiarism Analyzer]
    B --> D[AI Writing Detector]
    D --> E[AI‑Paraphrase Analyzer]
    C --> F[Sentence Highlighting]
    E --> F
    F --> G[Dashboard & Reports]
    G --> H[HTML / PDF Export]
```

---

## 🧠 Detection Capabilities

### AI Writing Detection
- Raw AI‑generated text
- GPT‑style probability smoothing
- Perplexity + burstiness metrics
- Transformer classifier support

### AI‑Paraphrased Detection
- Detects AI → paraphraser → output
- Synonym density analysis
- Structural instability signatures
- Operates only on AI‑flagged segments (low false positives)

### Plagiarism Detection
- Multi‑file similarity
- TF‑IDF + N‑gram overlap
- Sentence‑level plagiarism highlighting
- Cover & bibliography exclusion

---

## 🎨 Highlight Legend

| Color | Meaning |
|------|--------|
| 🔴 Red | Plagiarism |
| 🟠 Orange | AI‑Generated |
| 🟣 Purple | AI‑Generated + Paraphrased |

---

## 🖥 Running the Dashboard

```bash
py -3 AII.py flask
```

Open:
```
http://127.0.0.1:5000
```

---

## 🧪 Command‑Line Usage

Analyze documents:
```bash
py -3 AII.py file1.docx file2.pdf
```

Train AI classifier:
```bash
py -3 AII.py train_ai
```

Download external AI model:
```bash
py -3 AII.py download_model followsci/bert-ai-text-detector
```

---

## 🌍 GitHub Pages (Landing Page)

RedHydra is ready for **GitHub Pages**.

### Suggested setup
```
/docs
 ├── index.html
 ├── styles.css
 └── assets/
```

Use the README visuals + architecture diagram as your landing content.

---

## 🔐 Privacy & Ethics

- No cloud calls
- No telemetry
- No tracking
- Unlimited local use

> RedHydra is a **decision‑support system**, not an accusation engine.

---

## 🔗 Links

- GitHub: https://github.com/root60

---

<p align="center">
<b>RedHydra — engineered for trust, not fear.</b>
</p>
