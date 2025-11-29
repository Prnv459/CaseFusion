# CaseFusion
CaseFusion - AI Assistant for Cybercrime Investigators 
Overview
CaseFusion is an intelligent AI system that analyzes emails, phone call descriptions, bank statements, screenshots, and PDF files to identify potential scams and classify them into predefined categories. It uses multilingual NLP, rule-based entity extraction, and classification models to generate structured insights from unstructured case data.
Features :- 

1. Multi-modal Input Support
->Email text
->Phone call transcripts
->Bank statements
->Images (screenshots/photos)
->PDF documents
->Audio → converted to text

3. Entity Extraction
->Extracts key information:
Phone numbers
Email addresses
Bank details (account no., IFSC, UPI)
Names, locations, organisations
Monetary amounts
Scam-related keywords

4. Multilingual NLP
->Supports:
English
Hindi
Hinglish
Indian regional language variations

5. Classification Output
->Identifies categories such as:
Crypto Investment Scam
E-commerce Fraud
Impersonation Scam
UPI Scam / Bank Fraud
Email Spoofing

Architecture:-
       ┌────────────────────────┐
       │      User Input        │
       └────────────┬───────────┘
                    ▼
         ┌────────────────────┐
         │   Ingestion Layer  │
         └────────────┬───────┘
                      ▼
         ┌────────────────────-┐
         │    NLP Pipeline     │
         │ (NER + Keywords)    │
         └────────────┬────────┘
                      ▼
        ┌─────────────────────-┐
        │  Rule + Model Engine │
        └─────────────┬────────┘
                      ▼
       ┌─────────────────────────┐
       │ Structured JSON Output  │
       └─────────────────────────┘

CaseFusion(backend)/
│
├── ingestion.py
├── main.py
├── nlp.py
├── nlp_multilingual.py
├── schemas.py
├── graph_db.py
├── evaluate_casefusion.py
├── ground_truth.json
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

Evaluation Metrics:-
Input Type	   TP	    FP	   FN
   Phone	      2	    0	     0
   Email	      2	    0	     0
   Bank	        1	    0	     0
Overall Accuracy: 100%
CaseFusion successfully identified all the relevant entities and classified the scam correctly.

Use Cases:-
->🔍 Scam detection research
->🏦 Digital banking safety
->📧 Email fraud identification
->🧠 AI/ML academic projects
->🎓 NLP-based case analysis tools

Tech Stack:-
->Python 3.9+
->spaCy
->Transformers (HuggingFace)
->Google Generative AI (Gemini)
->regex / rule-based extraction
->PDF & image ingestion tools

License:-
This project is developed for academic purposes under institutional guidelines.

Contact:-
For queries or project evaluation: ranapranav912@gmail.com
