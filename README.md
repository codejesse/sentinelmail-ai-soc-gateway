# 🛡️ SentinelMail: AI Phishing Gateway & Agentic SOC Triage Platform

[![DevSecOps & MLOps CI](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=github-actions&logoColor=white)](#)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](#)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn_%7C_TensorFlow-F7931E?logo=scikit-learn&logoColor=white)](#)
[![LangChain](https://img.shields.io/badge/Agentic_AI-LangChain_%7C_Groq-1C3C3C?logo=chainlink&logoColor=white)](#)
[![Django](https://img.shields.io/badge/Dashboard-Django_5_%7C_Bootstrap_5-092E20?logo=django&logoColor=white)](#)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)](#)

**SentinelMail** is an end-to-end Machine Learning email security gateway, multi-class social-engineering classifier, and autonomous Security Operations Centre (SOC) triage platform. 

Built to solve Tier-1 SOC alert fatigue and manual review latency, SentinelMail combines a **Two-Stage ML Gatekeeper Pipeline** (Scikit-Learn, TensorFlow/Keras, SHAP, LIME) with an **Autonomous Tool-Calling LLM Triage Agent** (LangChain + Groq `openai/gpt-oss-120b`) and a real-time **Django Web Gateway & SOC Console**.


---

## 🌟 Key Capabilities

* **Stage 1 — Binary URL Gatekeeper (`task1.py`):** Evaluates structural URL and webpage indicators (`Dataset A`) to block or escalate links before inspecting message text. Includes cross-dataset target-leakage prevention, multicollinearity pruning (`> 0.90`), 5-fold Stratified Cross-Validation, a Keras MLP deep-learning baseline, and **SHAP (Global)** + **LIME (Local)** explainability.
* **Stage 2 — Multi-Class NLP Attack Classifier (`task2.py`):** Triggered conditionally only when Stage 1 flags a URL as malicious. Applies regex cleaning, stopword removal, and WordNet lemmatisation (`Dataset B`) to classify social-engineering techniques across **6 categories** (`Baiting`, `Malware`, `Phishing`, `Pretexting`, `Scareware`, and `NOT-Malicious General Class`) using sublinear TF-IDF bigrams (`LinearSVC`) benchmarked against a Keras `Embedding` neural network.
* **End-to-End CLI Pipeline (`task3_pipeline.py`):** Command-line gatekeeper utility that merges relational test streams (`Dataset C`), executes conditional Stage 1 $\rightarrow$ Stage 2 escalation, outputs `pipeline_verdict_report.csv`, and computes end-to-end SOC detection and error-propagation metrics.
* **Agentic SOC Triage Layer (`task4_agent.py`):** Autonomous LangChain agent (`create_agent`) powered by Groq (`openai/gpt-oss-120b`) equipped with `@tool` functions (`check_ip_reputation`, `get_user_logins`, `get_asset`). Defended against indirect prompt injection via `<untrusted_alert>` XML sandboxing.
* **Real-Time Django Email Gateway & SOC Console (`sentinelmail_web/`):** Full-stack web interface that ingests live simulated traffic or uploaded `.json`/`.csv` batches, automatically filters emails into **Safe Inbox**, **Warning Banner**, or **Quarantine Vault**, tags attack types, and renders live LangChain agent triage reports.
* **DevSecOps & MLOps CI/CD (`.github/workflows/`):** Automated GitHub Actions pipeline running `ruff` linting, `bandit` Static Application Security Testing (SAST), `pytest` unit/integration tests, and production `Docker` builds.

---

## 🏗️ System Architecture

```text
[Incoming Email Stream / JSON / CSV Upload / CLI Batch]
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│ STAGE 1: Binary URL Gatekeeper (task1.py)            │
│ • StandardScaler + Classical ML / Keras MLP          │
│ • SHAP Global Audit & LIME Per-Alert Explainability  │
└──────────────────────────┬───────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
      If Pred == 0                If Pred == 1
       (Benign URL)              (Malicious URL)
             │                           │
             ▼                           ▼
   [🟢 DELIVER TO INBOX]   ┌──────────────────────────────────────────────────────┐
   Tag: "NOT-Malicious"    │ STAGE 2: Multi-Class NLP Classifier (task2.py)       │
   (Bypasses Stage 2 &     │ • Regex Clean + Stopwords + WordNet Lemmatiser       │
    saves LLM compute)     │ • Sublinear TF-IDF (1,2-ngrams) + Balanced LinearSVC │
                           └──────────────────────────┬───────────────────────────┘
                                                      │
                                                      ▼
                                  [6-Class Social-Engineering Verdict]
                        (Baiting | Malware | Phishing | Pretexting | Scareware)
                                                      │
                                                      ▼
                           ┌──────────────────────────────────────────────────────┐
                           │ TASK 4: LangChain Autonomous SOC Agent (task4_agent) │
                           │ • Enforces <untrusted_alert> XML Prompt Sandboxing   │
                           │ • Calls @tool: check_ip_reputation(ip)               │
                           │ • Calls @tool: get_user_logins(username)             │
                           │ • Calls @tool: get_asset(hostname)                   │
                           └──────────────────────────┬───────────────────────────┘
                                                      │
                                                      ▼
                           ┌──────────────────────────────────────────────────────┐
                           │ DJANGO GATEWAY & SOC DASHBOARD (sentinelmail_web/)   │
                           │ • Auto-Routes: 🟢 Inbox | 🟡 Warning | 🔴 Quarantine │
                           │ • Displays Attack Badges & VERDICT: ESCALATE / CLOSE │
                           └──────────────────────────────────────────────────────┘

## GENERATING CUSTOM TEST DATASET TO TEST PIPELINE
# for older versions of python run python generate-random-dset.py --rows [specify row] --mode random --chaos [specifiy chaos range]
# e.g python generate-random-dset.py --rows 50 --mode random --chaos 1.8
# for python 3+ -> python3 generate-random-dset.py --rows 50 --mode random --chaos 1.8