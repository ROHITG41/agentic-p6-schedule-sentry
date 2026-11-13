# Autonomous Primavera P6 Schedule & Delay Sentry

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://rohit-kumar-p6-sentry.streamlit.app](https://rohit-kumar-p6-sentry.streamlit.app))
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Audit Standard](https://img.shields.io/badge/Audit%20Standard-DCMA%2014--Point-059669)
![Cognitive LLM](https://img.shields.io/badge/Cognitive%20LLM-gemini--3.8--flash-7C3AED)
![Architecture](https://img.shields.io/badge/Architecture-Deterministic--Cognitive%20Agent-1E3A8A)

> **Dataset & Confidentiality Note:** This application is an independent engineering architecture demonstration validated on representative, anonymized project datasets.

---

### Executive Problem Statement
On mega-infrastructure programs, manual Primavera P6 schedule reviews consume weeks of commercial and planning effort. Contractors frequently mask critical-path slippage through open-ended logic, negative lags, hard constraints, and artificial float hoarding—leading to protracted Extension of Time (EoT) disputes under FIDIC Sub-Clause 8.4.

### Operational Value
* **Accelerates Schedule Forensics:** Compresses baseline and monthly update integrity audits from manual spreadsheet cycles into an automated, repeatable workflow.
* **Exposes Logic Manipulation:** Deterministically flags missing predecessors/successors, negative leads, high-float anomalies, and Critical Path Length Index (CPLI) deviations prior to commercial review.
* **Defensible Claim Adjudication:** Grounds FIDIC Sub-Clause 8.4 delay assessments in verified CPU network metrics rather than subjective narrative claims.

---

### System Architecture

```mermaid
flowchart TD
    classDef client fill:#EFF6FF,stroke:#2563EB,stroke-width:2px,color:#1E3A8A;
    classDef deterministic fill:#ECFDF5,stroke:#059669,stroke-width:2px,color:#064E3B;
    classDef logic fill:#FAF5FF,stroke:#7C3AED,stroke-width:2px,color:#4C1D95;
    classDef output fill:#F8FAFC,stroke:#475569,stroke-width:2px,color:#0F172A;

    subgraph INGRESS ["1. SCHEDULE SNAPSHOT INGESTION"]
        UI["Streamlit Cloud Interface<br/><b>Primavera P6 Activity & Logic Network (.CSV / Sample)</b>"]:::client
    end

    subgraph CPU_GATE ["2. DETERMINISTIC CPU GRAPH & DCMA-14 ENGINE"]
        direction TB
        GRAPH["<b>Network Logic Traversal (CPU)</b><br/>Audits Open Ends, Negative Lags & Hard Constraints"]:::deterministic
        METRICS["<b>Schedule Health & Float Math</b><br/>Computes Total Float Distribution, CPLI & BEI"]:::deterministic
        CONTRACT["<b>Pydantic v2 Structured Payload</b><br/>Immutable Audit Scorecard"]:::deterministic
        GRAPH --> METRICS --> CONTRACT
    end

    subgraph COGNITIVE ["3. COGNITIVE DELAY ADJUDICATION LOOP"]
        LLM["<b>LangChain + Google Gemini (gemini-3.8-flash)</b><br/>Evaluates Critical-Path Slippage & Concurrency<br/>Applies FIDIC Sub-Clause 8.4 & SCL Delay Protocol"]:::logic
    end

    subgraph DELIVERABLE ["4. AUDITABLE FORENSIC OUTPUT"]
        MEMO["<b>Forensic Schedule Audit & EoT Determination Memo</b><br/>Clause-Backed Findings & Downloadable Report"]:::output
    end

    UI --> GRAPH
    CONTRACT -->|Verified Audit Metrics| LLM
    LLM --> MEMO
```

---

### Technology Stack
* **Runtime & UI:** Python 3.12, Streamlit Community Cloud
* **Orchestration & LLM:** LangChain, Google Gemini (`gemini-3.8-flash`)
* **Deterministic Analytics:** Pandas, NumPy, Pydantic v2

### Author & Architect
**Rohit Kumar, PMP, CEng (India), CEA, FSEr (TÜV)**  
Cross-Technology Technical Architect | Mission-Critical Infrastructure & Applied AI Solutions  
[LinkedIn](https://www.linkedin.com/in/rohitg41) | [Live Application](https://rohit-kumar-p6-sentry.streamlit.app)

---

### Enterprise Advisory & Custom Architecture
Available for technical consulting, custom agentic workflow development, and private enterprise deployments.  
**Connect:** [LinkedIn](https://www.linkedin.com/in/rohitg41) | **Email:** `grohit41@gmail.com`

---

### Intellectual Property & Licensing
**Copyright © 2026 Rohit Kumar. All Rights Reserved.**  
This repository and its source code are published solely for technical portfolio evaluation and architectural demonstration. Unauthorized copying, redistribution, or commercial deployment is prohibited without prior written consent.
