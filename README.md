---
title: P2P Anomaly Detection
---

# P2P Anomaly Detection with RAG-based Explanation

> Combining Machine Learning and Retrieval-Augmented Generation to make invoice fraud explainable in a Procure-to-Pay cycle.

This project builds an AI system that automatically detects suspicious vendor invoices in a Procure-to-Pay (P2P) process and explains why they were flagged. A machine learning model scans incoming invoices and scores them for risk. When a suspicious invoice is found, a RAG pipeline searches through procurement policy documents and vendor contracts to find relevant context, then uses an LLM to generate a plain-English explanation with a recommended action.

---

## Terminology

Understanding the domain language is key to understanding how this system works.

### Procure-to-Pay (P2P)
The end-to-end business process that covers everything from raising a purchase request to making a payment to a vendor. It includes: requisition → purchase order → goods receipt → invoice → payment. Fraud and errors typically occur at the invoice stage, which is what this system monitors.

### Purchase Order (PO)
A legally binding document a company sends to a vendor that authorises a specific purchase at an agreed price. Every invoice in this system is matched against a PO. If the invoice amount exceeds the PO amount, that is a red flag.

### Goods Receipt (GR)
A record confirming that ordered goods or services were actually received by the company's warehouse or operations team. A GR amount of $0 means nothing was received — if a vendor still invoices for it, that is a phantom delivery.

### Three-Way Match
The standard AP (Accounts Payable) control that checks three documents agree before a payment is approved:
- **Purchase Order** — what was authorised to be bought and at what price
- **Goods Receipt** — what was actually received
- **Invoice** — what the vendor is asking to be paid

All three must match within an acceptable tolerance. A failed three-way match is one of the strongest signals of a billing anomaly.

### Deviation %
The percentage difference between the invoice amount and the PO amount. Calculated as `((invoice_amount - po_amount) / po_amount) * 100`. AP policy in this system uses tiered thresholds: up to 5% is auto-approved, 5–15% requires manager sign-off, above 15% requires CFO escalation.

### AP (Accounts Payable)
The team or function responsible for processing and paying vendor invoices. They are the primary users of a system like this — the anomaly flags and RAG explanations are designed to support their review and escalation decisions.

### KYC (Know Your Vendor)
The due diligence process for verifying a new vendor's legitimacy before onboarding them and processing payments. New vendors with no transaction history requesting high-value invoices are flagged until KYC is complete.

### RAG (Retrieval-Augmented Generation)
An AI technique that grounds an LLM's response in retrieved documents rather than relying on its training data alone. In this system, when an anomaly is detected the RAG pipeline retrieves relevant AP policy documents, vendor contracts, and past confirmed fraud cases from a ChromaDB vector store, then passes them as context to GPT-4o to generate a grounded explanation.

### Vendor Contract
A document that specifies the agreed rate bands, payment terms, and service scope between the company and an established vendor. This system stores 10 vendor contracts in the RAG knowledge base — if an invoice deviates from the contracted rate, the explanation can cite the specific contract clause.

### Dispute Log
A record of a past anomalous invoice that was investigated, resolved, and documented. This system uses 800 synthetic dispute logs (200 per anomaly type) as RAG source documents — they give the LLM real precedent to cite when explaining why a current invoice looks suspicious.

---

## Anomaly Types Detected

| Anomaly | Trigger Condition | Risk |
|---|---|---|
| **Overbilling** | `invoice_amount > po_amount` | Vendor charging above the contracted rate |
| **Duplicate Invoice** | `days_since_last_invoice <= 7` | Same vendor re-submitting for the same PO within 7 days |
| **Phantom Delivery** | `gr_amount == 0` | Invoice raised for goods never received |
| **New Vendor Risk** | `is_new_vendor = true` and `invoice_amount > $10,000` | Unknown vendor requesting high-value payment with no history |

---

## Architecture

```
Invoice JSON
     │
     ▼
┌─────────────────────────────┐
│  POST /invoice (FastAPI)    │
│  ├── Field validation       │
│  ├── 4 rule-based detectors │
│  └── Random Forest scorer  │
└────────────┬────────────────┘
             │ anomaly flagged
             ▼
┌─────────────────────────────┐
│  POST /explain (FastAPI)    │
│  ├── ChromaDB retrieval     │
│  │   ├── Past fraud cases   │
│  │   ├── AP policy docs     │
│  │   └── Vendor contracts   │
│  └── GPT-4o explanation     │
└────────────┬────────────────┘
             │
             ▼
     Streamlit Demo UI
```

---

## Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| `invoice_id` | string | Unique invoice identifier |
| `vendor_id` | string | Vendor code (`V-001` = known, `V-NEW-XXX` = new) |
| `vendor_name` | string | Vendor display name |
| `vendor_category` | string | Category: `medical`, `industrial`, `IT`, `stationery`, `facilities`, `safety` |
| `po_reference` | string | Purchase Order reference number |
| `po_amount` | float | Pre-approved spend amount from PO |
| `invoice_amount` | float | Amount vendor is requesting payment for |
| `gr_amount` | float | Value of goods confirmed received (`0` = nothing received) |
| `deviation_pct` | float | % difference between invoice and PO amount |
| `days_since_last_invoice` | int | Days since vendor last submitted an invoice |
| `is_new_vendor` | int | `1` = no prior transaction history, `0` = known vendor |
| `three_way_match` | int | `1` = PO/GR/Invoice all agree, `0` = mismatch detected |
| `invoice_date` | string | Date invoice was submitted |
| `anomaly_type` | string | Human-readable anomaly label *(not used in training)* |
| `label` | int | Target variable — `0` = normal, `1` = anomaly |

---

## RAG Knowledge Base

| Document Type | Count | Purpose |
|---|---|---|
| Dispute logs | 800 | Past confirmed fraud cases with investigation notes (200 per anomaly type) |
| AP policy documents | 4 | Internal rules per anomaly type — thresholds, hold periods, escalation paths |
| Vendor contracts | 10 | Contracted rate bands and payment terms per established vendor |

---

## Project Structure

```
p2p-anomaly-detection/
├── api.py                        ← FastAPI app — /invoice and /explain endpoints
├── app.py                        ← Streamlit demo UI
├── rag.py                        ← RAGExplainer — ChromaDB retrieval + GPT-4o
├── train.py                      ← Random Forest training script
├── requirements.txt
├── data/
│   ├── p2p_invoices.csv          ← 20,000 row synthetic dataset
│   └── generate_p2p_data.py
├── documents/                    ← 814 source documents for RAG
├── models/
│   └── random_forest.joblib
├── interfaces/
│   └── base_detector.py
├── validators/
│   └── invoice_validator.py
├── detectors/
│   ├── overbilling_detector.py
│   ├── duplicate_detector.py
│   ├── phantom_delivery_detector.py
│   └── new_vendor_risk_detector.py
└── orchestrator/
    └── anomaly_orchestrator.py
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/p2p-anomaly-detection.git
cd p2p-anomaly-detection
pip install -r requirements.txt
```

### 2. Set environment variable

```bash
export OPENAI_API_KEY=your_key_here
```

### 3. Generate data and train the model

```bash
python data/generate_p2p_data.py
python train.py
```

### 4. Start the FastAPI backend

```bash
uvicorn api:app --reload
```

### 5. Launch the Streamlit UI

```bash
streamlit run app.py
```

---

## SOLID Principles Applied

| Principle | Implementation |
|---|---|
| **SRP** | Each class has one job — validate, detect one anomaly type, or orchestrate |
| **OCP** | Add a new detector by creating a new file — zero changes to existing code |
| **LSP** | All detectors extend `AnomalyDetector` ABC and are fully swappable |
| **ISP** | `RAGExplainer` is a separate class from the detectors |
| **DIP** | `AnomalyOrchestrator` depends on the `AnomalyDetector` abstraction, not concrete classes |
