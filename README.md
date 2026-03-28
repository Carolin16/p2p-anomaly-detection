# P2P Anomaly Detection with RAG-based Explanation

> Combining Machine Learning and Generative AI to make 
> invoice fraud explainable in a P2P cycle.

This project builds an AI system that automatically detects 
suspicious vendor invoices in a Procure-to-Pay (P2P) process 
and explains why they were flagged. A machine 
learning model scans incoming invoices and scores them for risk. 
When a suspicious invoice is found, a RAG (Retrieval-Augmented 
Generation) pipeline searches through procurement policy documents 
and vendor contracts to find relevant context, then uses an LLM 
to generate an explanation with a recommended action.

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

## Setup

### FAST API

1. pip install fastapi uvicorn
