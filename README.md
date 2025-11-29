<p align="center">
  <img src="https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/000000/brain-flaticons-lineal-color-flat-icons.png" width="70"/>
</p>

<h1 align="center">CaseFusion – AI-Powered Scam Case Identification System</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/NLP-Transformer%20Models-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-Academic-lightgrey?style=for-the-badge"/>
</p>

---

## 🚀 Overview

**CaseFusion** is an intelligent AI system that analyzes **emails**, **phone call descriptions**, **bank statements**, **screenshots**, and **PDF files** to detect potential **scams** and classify them into predefined categories.

It uses:
- Multilingual NLP  
- Entity extraction  
- Hybrid rule-based + AI classification  

to convert **unstructured case data** into **structured insights**.

---

## 🧠 Features

### ✔ Multi-modal Input Support
- 📧 Email text  
- 📞 Phone call transcripts  
- 🏦 Bank statements  
- 🖼 Images/screenshots  
- 📄 PDF documents  
- 🎙 Audio → converted to text  

### ✔ Entity Extraction
- Phone numbers  
- Email addresses  
- Bank details (Account no., IFSC, UPI)  
- Names, organisations  
- Amounts  
- Suspicious keywords  

### ✔ Multilingual NLP
Supports:
- English  
- Hindi  
- Hinglish  
- Indian regional variations  

### ✔ Classification Categories
- 🚨 Crypto Investment Scam  
- 🛒 E-commerce Fraud  
- 📞 Impersonation Fraud  
- 🏦 Bank Fraud / UPI Scam  
- 📧 Email Spoofing  

---

## 🏗 Architecture

```mermaid
flowchart TD
    A[User Input] --> B[Ingestion Layer]
    B --> C[NLP Pipeline (NER + Keywords)]
    C --> D[Rule + Model Engine]
    D --> E[Structured JSON Output]

CaseFusion/
│── ingestion.py
│── main.py
│── nlp.py
│── nlp_multilingual.py
│── schemas.py
│── graph_db.py
│── evaluate_casefusion.py
│── ground_truth.json
│
├── sample_cases/
│   ├── Case 1 mail.pdf
│   ├── case mail 2.pdf
│   ├── s1.jpeg
│   ├── s2.jpeg
│   ├── s3.jpeg
│   ├── s4.jpeg
│   └── s5.jpeg
│
└── evaluation/
    ├── recording_3.m4a
    └── results.json
