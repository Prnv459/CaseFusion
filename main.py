from collections import Counter
import re
import uuid
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from .schemas import (
    AnalysisResponse,
    ExtractedEntities,
    BankDetails,
)
from .ingestion import ingest_file
from .nlp import extract_cybercrime_entities, summarize_case
from .nlp_multilingual import extract_multilingual_entities
from .services.graph_db import graph_service
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="CaseFusion: Cybercrime Intelligence Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bank account utilities


def _clean_holder_name(name: str) -> str:
    """
    Clean noisy account holder names like 'BlueSky Enterprises Bank Name'
    -> 'BlueSky Enterprises'.
    """
    if not name:
        return "N/A"
    name = re.sub(r"\bBank Name\b", "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"\s+", " ", name)
    return name or "N/A"


def dedupe_bank_accounts(bank_accounts: List[BankDetails]) -> List[BankDetails]:
    """
    Merge duplicate bank accounts by account_number.
    Prefer the entry with richer info (non-'N/A' bank_name/ifsc/account_holder),
    and clean noisy holder names.
    """
    by_acc = {}

    for b in bank_accounts:
        acc = (b.account_number or "").strip()
        if not acc:
            continue

        # Normalise this record first
        b_bank = b.bank_name or "N/A"
        b_ifsc = b.ifsc or "N/A"
        b_holder = _clean_holder_name(b.account_holder)

        if acc not in by_acc:
            by_acc[acc] = BankDetails(
                account_number=acc,
                bank_name=b_bank,
                ifsc=b_ifsc,
                account_holder=b_holder,
            )
        else:
            existing = by_acc[acc]

            bank_name = existing.bank_name
            if (not bank_name or bank_name == "N/A") and b_bank != "N/A":
                bank_name = b_bank

            ifsc = existing.ifsc
            if (not ifsc or ifsc == "N/A") and b_ifsc != "N/A":
                ifsc = b_ifsc

            holder = existing.account_holder
            if (not holder or holder == "N/A") and b_holder != "N/A":
                holder = b_holder

            by_acc[acc] = BankDetails(
                account_number=acc,
                bank_name=bank_name or "N/A",
                ifsc=ifsc or "N/A",
                account_holder=_clean_holder_name(holder),
            )

    return list(by_acc.values())


def _union_clean(values: List[str]) -> List[str]:
    unique: List[str] = []
    for v in values:
        if not v or v in ("N/A", "Unknown"):
            continue
        if v not in unique:
            unique.append(v)
    return unique or ["N/A"]


def _levenshtein(a: str, b: str) -> int:
    a = a.lower()
    b = b.lower()
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            insert = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (ca != cb)
            curr.append(min(insert, delete, replace))
        prev = curr
    return prev[-1]


def _merge_similar_names(names: List[str], max_distance: int = 1) -> List[str]:
    if names == ["N/A"]:
        return names

    result: List[str] = []
    for name in names:
        placed = False
        for i, existing in enumerate(result):
            if _levenshtein(name, existing) <= max_distance:
                if len(name) < len(existing):
                    result[i] = name
                placed = True
                break
        if not placed:
            result.append(name)
    return result


def merge_case_entities(entities_list: List[ExtractedEntities]) -> ExtractedEntities:
    """
    Merge entities from multiple evidence files into a single case-level object.
    """
    if not entities_list:
        raise ValueError("No entities to merge")

    # Incident type: most common non-trivial label
    types = [
        e.incident_type
        for e in entities_list
        if e.incident_type and e.incident_type not in ("Unknown", "Extraction Failed")
    ]
    if types:
        incident_type = Counter(types).most_common(1)[0][0]
    else:
        incident_type = entities_list[0].incident_type

    victims = _union_clean([v for e in entities_list for v in e.victims])
    suspects = _union_clean(
        [s for e in entities_list for s in e.suspect_aliases])
    investigating_officers = _union_clean(
        [io for e in entities_list for io in e.investigating_officers]
    )

    victims = _merge_similar_names(victims)
    suspects = _merge_similar_names(suspects)

    phone_numbers = _union_clean(
        [p for e in entities_list for p in e.phone_numbers])
    email_addresses = _union_clean(
        [em for e in entities_list for em in e.email_addresses]
    )
    upi_ids = _union_clean([u for e in entities_list for u in e.upi_ids])
    ip_addresses = _union_clean(
        [ip for e in entities_list for ip in e.ip_addresses])
    fraudulent_urls = _union_clean(
        [u for e in entities_list for u in e.fraudulent_urls]
    )
    cryptocurrency_wallets = _union_clean(
        [w for e in entities_list for w in e.cryptocurrency_wallets]
    )

    all_banks: List[BankDetails] = []
    for e in entities_list:
        all_banks.extend(e.bank_accounts or [])
    bank_accounts = dedupe_bank_accounts(all_banks)

    # Amount lost: choose most frequent numeric amount
    amounts = []
    for e in entities_list:
        if not e.amount_lost or e.amount_lost == "N/A":
            continue
        m = re.search(r"([\d,]+(?:\.\d+)?)", e.amount_lost)
        if not m:
            continue
        val_str = m.group(1).replace(",", "")
        try:
            val = float(val_str)
            amounts.append(val)
        except ValueError:
            continue

    if amounts:
        val, _ = Counter(amounts).most_common(1)[0]
        amount_lost = f"INR {val:,.2f}"
    else:
        amount_lost = "N/A"

    return ExtractedEntities(
        case_number="N/A",
        incident_type=incident_type,
        victims=victims,
        suspect_aliases=suspects,
        investigating_officers=investigating_officers,
        phone_numbers=phone_numbers,
        email_addresses=email_addresses,
        upi_ids=upi_ids,
        bank_accounts=bank_accounts,
        ip_addresses=ip_addresses,
        fraudulent_urls=fraudulent_urls,
        cryptocurrency_wallets=cryptocurrency_wallets,
        amount_lost=amount_lost,
    )


# --------------------------------------------------------
# USE CASE 1 & 2: core cyber report (multi-file)
# --------------------------------------------------------


@app.post("/analyze_cyber_report", response_model=AnalysisResponse)
def process_cyber_report(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # 1️⃣ Ingest all files in parallel (faster if multiple PDFs/emails)
    with ThreadPoolExecutor(max_workers=min(4, len(files))) as pool:
        evidence_docs = list(pool.map(ingest_file, files))

        # 2️⃣ Extract entities for each evidence in parallel as well
        def _extract_for_doc(ed):
            return extract_cybercrime_entities(
                ed.raw_text,
                evidence_type=ed.evidence_type,
            )

        entities_list: List[ExtractedEntities] = list(
            pool.map(_extract_for_doc, evidence_docs))

    # 3️⃣ Merge entities across all evidence docs
    merged_entities = merge_case_entities(entities_list)

    # 4️⃣ Build graph entities
    case_id = str(uuid.uuid4())

    graph_entities = []
    entity_map = {
        "PHONE": merged_entities.phone_numbers,
        "EMAIL": merged_entities.email_addresses,
        "UPI": merged_entities.upi_ids,
        "IP_ADDRESS": merged_entities.ip_addresses,
        "URL": merged_entities.fraudulent_urls,
        "WALLET": merged_entities.cryptocurrency_wallets,
        "SUSPECT": merged_entities.suspect_aliases,
        "VICTIM": merged_entities.victims,
    }

    for entity_type, value_list in entity_map.items():
        for value in value_list:
            if value and value not in ("N/A", "Unknown"):
                graph_entities.append({"type": entity_type, "value": value})

    for bank in merged_entities.bank_accounts:
        if bank.account_number and bank.account_number != "N/A":
            display_val = bank.account_number
            if bank.bank_name and bank.bank_name != "N/A":
                display_val += f" ({bank.bank_name})"
            graph_entities.append(
                {"type": "BANK_ACCOUNT", "value": display_val})

    # 5️⃣ Write to graph DB (same as before)
    graph_result = graph_service.ingest_case_data(
        case_id=case_id,
        case_title=f"Case with {len(evidence_docs)} file(s)",
        entities=graph_entities,
    )

    # 6️⃣ Build combined text once for summarization
    combined_text_parts = [
        f"--- FILE: {ed.filename} ---\n{ed.raw_text}"
        for ed in evidence_docs
    ]
    combined_text = "\n\n".join(combined_text_parts)

    # 🔹 Optional (safe) cap on text length for speed, without hurting accuracy much.
    # You can increase/decrease MAX_SUMMARY_CHARS if needed.
    MAX_SUMMARY_CHARS = 15000
    summary_input = (
        combined_text
        if len(combined_text) <= MAX_SUMMARY_CHARS
        else combined_text[:MAX_SUMMARY_CHARS]
    )

    summary = summarize_case(summary_input, merged_entities)

    # 7️⃣ File meta for response
    filenames = [ed.filename for ed in evidence_docs]
    content_types = [ed.content_type for ed in evidence_docs]

    if len(filenames) == 1:
        display_filename = filenames[0]
        display_content_type = content_types[0]
    else:
        display_filename = f"{filenames[0]} (+{len(filenames) - 1} more)"
        display_content_type = "multiple"

    return AnalysisResponse(
        filename=display_filename,
        content_type=display_content_type,
        filenames=filenames,
        content_types=content_types,
        extracted_entities=merged_entities,
        case_id=case_id,
        alerts=graph_result.get("alerts", []),
        summary=summary,
    )


# --------------------------------------------------------
# USE CASE 3: Multilingual (single-file for now)
# --------------------------------------------------------


@app.post("/analyze_multilingual_report", response_model=AnalysisResponse)
def process_multilingual_report(file: UploadFile = File(...)):
    evidence_doc = ingest_file(file)
    text = evidence_doc.raw_text

    entities = extract_multilingual_entities(text)

    # If multilingual pipeline returns BankDetails objects, dedupe them
    if hasattr(entities, "bank_accounts"):
        try:
            entities.bank_accounts = dedupe_bank_accounts(
                entities.bank_accounts)
        except Exception:
            pass

    case_id = str(uuid.uuid4())

    graph_entities = []
    entity_map = {
        "PHONE": entities.phone_numbers,
        "EMAIL": entities.email_addresses,
        "UPI": entities.upi_ids,
        "IP_ADDRESS": entities.ip_addresses,
    }
    for entity_type, value_list in entity_map.items():
        for value in value_list:
            if value and value not in ("N/A", "Unknown"):
                graph_entities.append({"type": entity_type, "value": value})

    if hasattr(entities, "bank_accounts"):
        for bank in entities.bank_accounts:
            if hasattr(bank, "account_number") and bank.account_number:
                display_val = bank.account_number
                if getattr(bank, "bank_name", "N/A") not in (None, "N/A"):
                    display_val += f" ({bank.bank_name})"
                graph_entities.append(
                    {"type": "BANK_ACCOUNT", "value": display_val})
            else:
                graph_entities.append(
                    {"type": "BANK_ACCOUNT", "value": str(bank)})

    graph_result = graph_service.ingest_case_data(
        case_id=case_id,
        case_title=f"Multilingual Case: {file.filename}",
        entities=graph_entities,
    )

    summary = summarize_case(text, entities)

    return AnalysisResponse(
        filename=file.filename,
        content_type=file.content_type,
        filenames=[file.filename],
        content_types=[file.content_type],
        extracted_entities=entities,
        case_id=case_id,
        alerts=graph_result.get("alerts", []),
        summary=summary,
    )


# --------------------------------------------------------
# Graph visualisation endpoint
# --------------------------------------------------------


@app.get("/case/{case_id}/network")
def get_case_network_graph(case_id: str):
    try:
        network_data = graph_service.get_case_network(case_id)
        if not network_data["nodes"]:
            raise HTTPException(
                status_code=404, detail="Case not found or has no entities"
            )
        return network_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------
# Old test endpoint + root
# --------------------------------------------------------


class EntityItem(BaseModel):
    type: str
    value: str


class CaseRequest(BaseModel):
    case_id: str
    case_title: str
    extracted_entities: List[EntityItem]


@app.post("/analyze-network")
def analyze_network(data: CaseRequest):
    entities_list = [{"type": e.type, "value": e.value}
                     for e in data.extracted_entities]
    result = graph_service.ingest_case_data(
        case_id=data.case_id,
        case_title=data.case_title,
        entities=entities_list,
    )
    return result


@app.get("/")
def read_root():
    return {"message": "CaseFusion Backend is Running"}
