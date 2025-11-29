from pydantic import BaseModel
from typing import List, Optional, Any

# 0. INGESTION STRUCTURES (used by ingest_file)


class PageText(BaseModel):
    page_number: int
    text: str


class EvidenceDoc(BaseModel):
    filename: str
    content_type: str
    raw_text: str
    pages: List[PageText]
    evidence_type: str  # "chat", "email", "bank_document", "generic"


# 1. BANK DETAILS


class BankDetails(BaseModel):
    account_number: str
    bank_name: str = "N/A"
    ifsc: str = "N/A"
    account_holder: str = "N/A"


# 2. EXTRACTED ENTITIES (output of extraction pipeline)


class ExtractedEntities(BaseModel):
    case_number: str
    incident_type: str
    victims: List[str]
    suspect_aliases: List[str]
    investigating_officers: List[str]
    phone_numbers: List[str]
    email_addresses: List[str]
    upi_ids: List[str]
    bank_accounts: List[BankDetails]
    ip_addresses: List[str]
    fraudulent_urls: List[str]
    cryptocurrency_wallets: List[str]
    amount_lost: str


# 3. FINAL API RESPONSE


class AnalysisResponse(BaseModel):
    # Backward-compatible single-file display
    filename: str
    content_type: str

    # NEW: full list when multiple files are uploaded
    filenames: Optional[List[str]] = None
    content_types: Optional[List[str]] = None

    extracted_entities: ExtractedEntities
    case_id: str
    alerts: List[Any] = []
    summary: str = ""
