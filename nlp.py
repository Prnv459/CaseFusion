import os
import re
import json
from typing import List, Tuple
from urllib.parse import urlparse

import spacy
from spacy.pipeline import EntityRuler
from transformers import pipeline
import google.generativeai as genai

from .schemas import ExtractedEntities, BankDetails

# 0. GOOGLE / GEMINI CONFIG

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
else:
    gemini_model = None
    print(
        "[CaseFusion] WARNING: GOOGLE_API_KEY not set. "
        "Gemini summary (and optional LLM extraction) will be skipped."
    )

# 1. Load spaCy + Zero-shot Classifier

try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    print("[CaseFusion] Transformer model not found. Falling back to en_core_web_sm.")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download

        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

# EntityRuler for IFSC / bank account / UPI patterns
if "entity_ruler" in nlp.pipe_names:
    ruler = nlp.get_pipe("entity_ruler")
else:
    if "ner" in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.add_pipe("entity_ruler")

ENTITY_PATTERNS = [
    {"label": "IFSC", "pattern": [
        {"TEXT": {"REGEX": r"^[A-Z]{4}0[A-Z0-9]{6}$"}}]},
    {"label": "BANK_ACC", "pattern": [{"TEXT": {"REGEX": r"^\d{9,18}$"}}]},
    {"label": "UPI_ID", "pattern": [
        {"TEXT": {"REGEX": r"^[a-zA-Z0-9.\-_]+@[a-zA-Z0-9]+$"}}]},
]
ruler.add_patterns(ENTITY_PATTERNS)

# Zero-shot classifier for incident type
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)

UPI_HANDLES = {
    "oksbi",
    "okaxis",
    "okhdfcbank",
    "okicici",
    "ybl",
    "ibl",
    "upi",
    "axl",
    "paytm",
    "apl",
    "sbi",
    "idfcbank",
    "kotak",
    "payzapp",
    "okyesbank",
}

# Common real banks we don't want as suspects by default
BANK_ORG_STOP = {
    "hdfc",
    "hdfc bank",
    "state bank of india",
    "sbi",
    "icici bank",
    "icici",
    "axis bank",
    "axis",
    "kotak mahindra bank",
    "kotak",
    "bank of baroda",
}

# Cities / locations that shouldn't be suspects by themselves
CITY_STOP = {
    "delhi",
    "dubai",
    "london",
    "singapore",
    "mumbai",
    "bangalore",
    "bengaluru",
    "noida",
    "gurgaon",
    "pune",
    "hyderabad",
    "chennai",
    "kolkata",
    "india",
}

# Generic junk tokens we don't want as suspects
GENERIC_BAD_SUSPECTS = {
    "kyc",
    "k y c",
    "upi",
    "salary",
    "task",
    "gateway",
    "merchant gateway",
    "merchant",
    "security deposit",
    "deposit",
}


# 2. Common helpers

def clean_text_artifacts(text: str) -> str:
    text = re.sub(r"Page \d+ of \d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def classify_incident_intelligently(text: str) -> str:
    narrative_snippet = text[:1000]
    candidate_labels = [
        "Crypto Investment Scam",
        "Job Task Fraud",
        "Task/Salary Scam",
        "Phishing",
        "Sextortion",
        "Loan App Harassment",
        "Identity Theft",
        "Social Media Harassment",
        "Generic Cyber Fraud",
    ]
    result = classifier(narrative_snippet, candidate_labels, multi_label=False)
    return result["labels"][0]


def _list_is_valid(lst: List[str]) -> bool:
    if not lst:
        return False
    if len(lst) == 1 and lst[0] in ("N/A", "Unknown", "", None):
        return False
    return True


def _looks_like_mobile_with_country_code(num: str) -> bool:
    """
    e.g. 918289002887 (91 + 8289002887)
    """
    num = num.strip()
    if not num.isdigit():
        return False
    if num.startswith("91") and len(num) in (11, 12, 13):
        last10 = num[-10:]
        if len(last10) == 10 and last10[0] in "6789":
            return True
    return False


def _merge_list(preferred: List[str], backup: List[str]) -> List[str]:
    if _list_is_valid(preferred):
        return sorted(set(preferred))
    if _list_is_valid(backup):
        return sorted(set(backup))
    return ["N/A"]


def _normalize_name_for_role(raw: str) -> str:
    """
    Clean raw PERSON/ORG text so we don't keep things like:
    'Amit Verma <email> 22 November 2025 at 15:07'
    """
    if not raw:
        return ""

    n = raw

    # Remove email chunks in angle brackets
    n = re.sub(r"<[^>]+>", "", n)

    # Remove trailing date + time patterns like '22 November 2025 at 15:07'
    n = re.sub(
        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}.*$",
        "",
        n,
        flags=re.IGNORECASE,
    )

    # Remove standalone time tokens like '3:22'
    n = re.sub(r"\b\d{1,2}:\d{2}\b", "", n)

    # Collapse spaces and strip punctuation
    n = re.sub(r"\s+", " ", n).strip(" ,;:-")
    if not n:
        return ""

    # If name still contains digits or '@', it's probably junk
    if any(ch.isdigit() for ch in n) or "@" in n:
        return ""

    return n.strip()


def _clean_suspect_aliases(names: List[str]) -> List[str]:
    """
    Post-process suspect list:
    - remove junk like 'UPI', 'KYC'
    - drop pure bank names / bare city names
    - clean 'BlueSky Enterprises Bank Name' -> 'BlueSky Enterprises'
    """
    cleaned: List[str] = []
    for n in names:
        if not n or n in ("N/A", "Unknown"):
            continue

        # remove 'Bank Name' literal
        n2 = re.sub(r"\bBank Name\b", "", n, flags=re.IGNORECASE).strip()
        # collapse repeated spaces
        n2 = re.sub(r"\s+", " ", n2)
        low = n2.lower()

        # throw away obviously non-actor tokens
        if (
            low in GENERIC_BAD_SUSPECTS
            or low in BANK_ORG_STOP
            or low in CITY_STOP
        ):
            continue

        # special: if something like "BlueSky Enterprises Bank", trim trailing 'Bank'
        if re.search(r"\bBank\b$", n2):
            n2 = re.sub(r"\bBank\b$", "", n2).strip()

        if n2 and n2 not in cleaned:
            cleaned.append(n2)

    return cleaned or ["N/A"]


# 3. RULE-BASED ENGINE (spaCy + regex)


class ContextBankParser:
    BANK_LINE_PATTERN = re.compile(
        r"""
        (?P<label>(?:Acc(?:ount)?\s*(?:No|Number)?|A/C\s*No\.?))\s*[:\-]?\s*
        (?P<acc>\d{9,18})[^\n]*?
        (?:IFSC\s*(?:Code)?\s*[:\-]?\s*(?P<ifsc>[A-Z]{4}0[A-Z0-9]{6}))?
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def __init__(self, text: str):
        self.lines = [line.strip()
                      for line in text.split("\n") if line.strip()]
        self.raw_text = text

    def extract(self) -> List[BankDetails]:
        found_banks: List[BankDetails] = []
        current_bank = {}

        patterns = {
            "acc_no": r"(?:Acc(?:ount)?|A/C)\s*(?:No|Number)?\.?\s*[:\-]?\s*(\d{9,18})",
            "ifsc": r"(?:IFSC)\s*(?:Code)?\s*[:\-]?\s*([A-Z]{4}0[A-Z0-9]{6})",
            "bank_name": r"(?:Bank)\s*(?:Name)?\s*[:\-]?\s*([A-Za-z\s]+(?:Bank|Ltd))",
            "holder": r"(?:Account|Beneficiary)\s*Name\s*[:\-]?\s*([A-Za-z\s]+)",
        }

        # 1) High-confidence combined pattern on single lines
        for line in self.lines:
            m = self.BANK_LINE_PATTERN.search(line)
            if m:
                acc = m.group("acc")
                ifsc = m.group("ifsc") or "N/A"
                found_banks.append(
                    BankDetails(
                        account_number=acc,
                        ifsc=ifsc,
                    )
                )

        # 2) Incremental grouping across lines
        for line in self.lines:
            acc_match = re.search(patterns["acc_no"], line, re.IGNORECASE)
            if acc_match:
                if "account_number" in current_bank:
                    found_banks.append(BankDetails(**current_bank))
                    current_bank = {}
                current_bank["account_number"] = acc_match.group(1)

            ifsc_match = re.search(patterns["ifsc"], line, re.IGNORECASE)
            if ifsc_match:
                current_bank["ifsc"] = ifsc_match.group(1)

            bank_match = re.search(patterns["bank_name"], line, re.IGNORECASE)
            if bank_match:
                current_bank["bank_name"] = bank_match.group(1).strip()

            holder_match = re.search(patterns["holder"], line, re.IGNORECASE)
            if holder_match:
                holder = holder_match.group(1)
                holder = re.sub(r"\bBank Name\b", "", holder,
                                flags=re.IGNORECASE).strip()
                current_bank["account_holder"] = holder

        if "account_number" in current_bank:
            found_banks.append(BankDetails(**current_bank))

        # 3) Fallback: orphan numbers that look like bank accounts
        if not found_banks:
            raw_numbers = re.findall(r"\b\d{9,18}\b", self.raw_text)
            clean_numbers = []
            for n in raw_numbers:
                if len(n) == 10 and n[0] in "6789":
                    continue
                if _looks_like_mobile_with_country_code(n):
                    continue
                clean_numbers.append(n)

            for n in clean_numbers:
                found_banks.append(BankDetails(account_number=n))

        return found_banks


class RoleAssigner:
    def __init__(self, text: str, all_names: List[str], evidence_type: str):
        self.text = text
        self.names = all_names
        self.evidence_type = evidence_type

    def _extract_email_headers(self) -> Tuple[List[str], List[str]]:
        """
        Very simple header parser for From: and To: lines in email-like text.
        Returns (to_names, from_names).
        """
        t = self.text

        to_names: List[str] = []
        from_names: List[str] = []

        # Match lines starting with "To:" or "From:"
        to_match = re.findall(r"^To:\s*(.+)$", t,
                              flags=re.IGNORECASE | re.MULTILINE)
        from_match = re.findall(r"^From:\s*(.+)$", t,
                                flags=re.IGNORECASE | re.MULTILINE)

        def clean_header_line(line: str) -> str:
            # Remove any email in <...>
            line = re.sub(r"<[^>]+>", "", line)
            # Remove quotes and extra spaces
            line = line.replace('"', "").strip()
            return line

        for line in to_match:
            name = _normalize_name_for_role(clean_header_line(line))
            # Ignore pure emails or empty
            if not name:
                continue
            to_names.append(name)

        for line in from_match:
            name = _normalize_name_for_role(clean_header_line(line))
            if not name:
                continue
            from_names.append(name)

        return to_names, from_names

    def assign(self) -> Tuple[List[str], List[str]]:
        victims = set()
        suspects = set()
        t = self.text

        # --- SPECIAL LOGIC FOR EMAILS ---
        if self.evidence_type == "email":
            # 1) Use explicit headers if present
            to_names, from_names = self._extract_email_headers()
            for n in to_names:
                victims.add(n)
            for n in from_names:
                suspects.add(n)

            # 1b) Gmail-style header: 'Pranav Rana <email>' (victim)
            if not victims:
                header_match = re.search(
                    r"^([A-Z][A-Za-z ]{2,60})\s*<[^>]+>",
                    t,
                    flags=re.MULTILINE,
                )
                if header_match:
                    h_name = _normalize_name_for_role(header_match.group(1))
                    if h_name:
                        victims.add(h_name)

            # 2) Use "Dear <name>" / closing patterns as additional hints
            for name in self.names:
                norm_name = _normalize_name_for_role(name)
                if not norm_name:
                    continue

                if re.search(
                    rf"Dear\s+.*?\b{re.escape(norm_name)}\b",
                    t,
                    re.IGNORECASE,
                ):
                    victims.add(norm_name)
                elif re.search(
                    rf"(Regards|Sincerely|Thanks)[\s,:]*\n*\s*{re.escape(norm_name)}",
                    t,
                    re.IGNORECASE,
                ):
                    suspects.add(norm_name)
                elif re.search(
                    rf"From:.*?\b{re.escape(norm_name)}\b",
                    t,
                    re.IGNORECASE,
                ):
                    suspects.add(norm_name)

        # --- LOGIC FOR CHAT SCREENSHOTS ---
        elif self.evidence_type == "chat":
            # Sometimes exported chats have "Chat with X"
            header_match = re.search(
                r"Chat with\s+([A-Za-z0-9\s]+)", t, re.IGNORECASE
            )
            if header_match:
                h_name = _normalize_name_for_role(header_match.group(1))
                if h_name:
                    suspects.add(h_name)

            # Greeting: "Hello Pranav" -> victim = Pranav
            greet_match = re.search(
                r"\b(?:Hello|Hi|Dear)\s+([A-Z][A-Za-z]{2,40})",
                t,
                flags=re.IGNORECASE,
            )
            if greet_match:
                v_name = _normalize_name_for_role(greet_match.group(1))
                if v_name:
                    victims.add(v_name)

            # Intro: "this is Amit Verma from ..." -> suspect = Amit Verma
            intro_match = re.search(
                r"\bthis is\s+([A-Z][A-Za-z ]{2,60})\s+from",
                t,
                flags=re.IGNORECASE,
            )
            if intro_match:
                s_name = _normalize_name_for_role(intro_match.group(1))
                if s_name:
                    suspects.add(s_name)

            # Commands: 'pay/transfer/deposit to <Name>'
            for name in self.names:
                norm_name = _normalize_name_for_role(name)
                if not norm_name:
                    continue

                if re.search(
                    rf"(?:pay|transfer|send|deposit)\s+(?:to\s+)?{re.escape(norm_name)}",
                    t,
                    re.IGNORECASE,
                ):
                    suspects.add(norm_name)

                if re.search(
                    rf"{re.escape(norm_name)}[:\-\s].*(?:send|pay|transfer|deposit).*",
                    t,
                    re.IGNORECASE,
                ):
                    suspects.add(norm_name)

        # --- FALLBACK / GENERIC ---
        if not victims and not suspects:
            for name in self.names:
                norm_name = _normalize_name_for_role(name)
                if not norm_name:
                    continue

                if re.search(rf"Dear\s+{re.escape(norm_name)}", t, re.IGNORECASE):
                    victims.add(norm_name)
                elif re.search(
                    rf"From:.*?{re.escape(norm_name)}", t, re.IGNORECASE
                ):
                    suspects.add(norm_name)

        suspects = suspects - victims
        return sorted(victims), sorted(suspects)


class UniversalParser:
    def __init__(self, text: str, evidence_type: str):
        self.raw_text = text
        self.clean_text = clean_text_artifacts(text)
        self.evidence_type = evidence_type
        self.doc = nlp(self.clean_text)

    def extract_persons_orgs(self) -> Tuple[List[str], List[str]]:
        persons = set()
        orgs = set()
        for ent in self.doc.ents:
            if ent.label_ == "PERSON":
                name_raw = ent.text.strip()
                name = _normalize_name_for_role(name_raw)
                if (
                    name
                    and len(name) > 2
                    and name.lower()
                    not in {
                        "user",
                        "customer",
                        "admin",
                        "manager",
                        "sir",
                        "madam",
                        "valued investor",
                        "employee",
                    }
                ):
                    persons.add(name)
            elif ent.label_ in ("ORG", "GPE"):
                org_raw = ent.text.strip()
                org = _normalize_name_for_role(org_raw)
                if org and len(org) > 2:
                    orgs.add(org)
        return list(persons), list(orgs)

    def extract_amount(self) -> str:
        """
        Try to identify the actual loss / deposit amount rather than
        salary or pending payout.
        """
        text = self.clean_text
        amount_pattern = r"(?:INR|Rs\.?|₹)\s*([\d,]+(?:\.\d{1,2})?)"
        matches = []
        for m in re.finditer(amount_pattern, text, flags=re.IGNORECASE):
            val_str = m.group(1).replace(",", "")
            try:
                val = float(val_str)
            except ValueError:
                continue

            # Ignore tiny values (spurious OCR like '2.')
            if val < 100:
                continue

            start = max(0, m.start() - 80)
            end = min(len(text), m.end() + 80)
            ctx = text[start:end].lower()
            matches.append((val, ctx))

        if not matches:
            return "N/A"

        scores = []
        for val, ctx in matches:
            score = 1.0
            if any(w in ctx for w in ["deposit", "refundable", "security", "fee", "charge", "verification"]):
                score += 3
            if any(w in ctx for w in ["transfer", "transferred", "sent", "paid", "pay", "debit"]):
                score += 2
            if any(w in ctx for w in ["salary", "payout", "pending amount", "pending", "credit", "credited"]):
                score -= 2
            scores.append((score, val))

        scores.sort(reverse=True)
        best_val = scores[0][1]
        return f"INR {best_val:,.2f}"

    def extract_phones(self) -> List[str]:
        raw_phones = re.findall(
            r"(?:\+91[\-\s]?)?([6-9]\d{4}[\-\s]?\d{5})", self.clean_text
        )
        clean_phones = {p.replace(" ", "").replace("-", "")
                        for p in raw_phones}
        clean_phones = [p for p in clean_phones if len(p) == 10]
        return sorted(clean_phones)

    def extract_emails(self) -> List[str]:
        emails = list(
            set(
                re.findall(
                    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
                    self.clean_text,
                )
            )
        )
        return emails

    def extract_upis(self, emails: List[str]) -> List[str]:
        raw_upis = re.findall(
            r"\b[a-zA-Z0-9.\-_]+@[a-zA-Z0-9]+\b", self.clean_text)
        email_set = set(emails)
        clean_upis = set()
        for upi in raw_upis:
            if upi in email_set:
                continue
            handle = upi.split("@")[1].lower()
            if handle in UPI_HANDLES:
                clean_upis.add(upi)
        return sorted(clean_upis)

    def extract_urls(self) -> List[str]:
        raw_urls = re.findall(r"https?://[^\s<>'\"()]+", self.clean_text)
        clean_urls = set()
        for url in raw_urls:
            parsed = urlparse(url)
            if "google.com" in parsed.netloc and "forms" not in parsed.path:
                continue
            clean_urls.add(url)
        return sorted(clean_urls)

    def extract_ips(self) -> List[str]:
        ips = list(
            set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", self.clean_text)))
        return ips

    def process(self) -> ExtractedEntities:
        bank_parser = ContextBankParser(self.clean_text)
        banks = bank_parser.extract()

        persons, orgs = self.extract_persons_orgs()
        role_assigner = RoleAssigner(
            self.clean_text, persons, self.evidence_type)
        victims, person_suspects = role_assigner.assign()

        # Organisations + person suspects => raw suspect list
        all_suspects_raw = sorted(set(person_suspects) | set(orgs))
        all_suspects = _clean_suspect_aliases(all_suspects_raw)

        amount = self.extract_amount()
        incident = classify_incident_intelligently(self.clean_text)

        phones = self.extract_phones()
        emails = self.extract_emails()
        upis = self.extract_upis(emails)
        urls = self.extract_urls()
        ips = self.extract_ips()

        return ExtractedEntities(
            case_number="N/A",
            incident_type=incident,
            victims=victims or ["Unknown"],
            suspect_aliases=all_suspects or ["Unknown"],
            investigating_officers=["N/A"],
            phone_numbers=phones or ["N/A"],
            email_addresses=emails or ["N/A"],
            upi_ids=upis or ["N/A"],
            bank_accounts=banks,
            ip_addresses=ips or ["N/A"],
            fraudulent_urls=urls or ["N/A"],
            cryptocurrency_wallets=["N/A"],
            amount_lost=amount,
        )


def extract_with_rules(text: str, evidence_type: str = "generic") -> ExtractedEntities:
    parser = UniversalParser(text, evidence_type)
    return parser.process()


# 4. OPTIONAL GEMINI-BASED EXTRACTION (JSON)


def clean_json_response(text: str) -> str:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return text.strip()


def extract_with_gemini(text: str, evidence_type: str = "generic") -> ExtractedEntities:
    if gemini_model is None:
        return ExtractedEntities(
            case_number="N/A",
            incident_type="Extraction Failed",
            victims=[],
            suspect_aliases=[],
            investigating_officers=[],
            phone_numbers=[],
            email_addresses=[],
            upi_ids=[],
            bank_accounts=[],
            ip_addresses=[],
            fraudulent_urls=[],
            cryptocurrency_wallets=[],
            amount_lost="N/A",
        )

    prompt = f"""
    You are a specialized Cybercrime Investigator. Extract structured evidence from this {evidence_type}.
    
    STRICT EXTRACTION RULES:
    1. Victim: The person receiving the threat/scam or losing money.
    2. Suspect: The sender, or person demanding money/OTP.
    3. Banks: Group Account Numbers with their Bank Names/IFSC.
    4. Amounts: Only the fraud/demanded amount or total loss.
    5. UPI: Ignore normal emails. Only payment handles (e.g., user@bank).
    6. Missing Data: If a field is not found, return null or [].

    OUTPUT SCHEMA (Valid JSON Only):
    {{
        "incident_type": "String",
        "victims": ["Name1"],
        "suspect_aliases": ["Name1"],
        "phone_numbers": ["Str"],
        "email_addresses": ["Str"],
        "upi_ids": ["Str"],
        "bank_accounts": [
            {{
                "account_number": "Str",
                "bank_name": "Str",
                "ifsc": "Str",
                "account_holder": "Str"
            }}
        ],
        "fraudulent_urls": ["Str"],
        "amount_lost": "Str"
    }}

    EVIDENCE TEXT:
    {text}
    """

    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            },
        )
        if not response.text:
            raise ValueError("Empty response from Gemini")

        data = json.loads(clean_json_response(response.text))

        return ExtractedEntities(
            case_number="N/A",
            incident_type=data.get("incident_type", "Unknown"),
            victims=data.get("victims") or [],
            suspect_aliases=data.get("suspect_aliases") or [],
            investigating_officers=[],
            phone_numbers=data.get("phone_numbers") or [],
            email_addresses=data.get("email_addresses") or [],
            upi_ids=data.get("upi_ids") or [],
            bank_accounts=[BankDetails(**b)
                           for b in data.get("bank_accounts", [])],
            ip_addresses=[],
            fraudulent_urls=data.get("fraudulent_urls") or [],
            cryptocurrency_wallets=[],
            amount_lost=data.get("amount_lost") or "N/A",
        )
    except Exception as e:
        print(f"[CaseFusion] ❌ Gemini extraction error: {e}")
        return ExtractedEntities(
            case_number="N/A",
            incident_type="Extraction Failed",
            victims=[],
            suspect_aliases=[],
            investigating_officers=[],
            phone_numbers=[],
            email_addresses=[],
            upi_ids=[],
            bank_accounts=[],
            ip_addresses=[],
            fraudulent_urls=[],
            cryptocurrency_wallets=[],
            amount_lost="N/A",
        )


# 5. HYBRID MERGE ENGINE (OPTIONAL)


def _clean_suspects(suspects: List[str]) -> List[str]:
    """
    For hybrid (rules + LLM) path, just reuse the same cleaning
    logic as _clean_suspect_aliases so behaviour stays consistent.
    """
    return _clean_suspect_aliases(suspects)


def _merge_entities(rules: ExtractedEntities, llm: ExtractedEntities) -> ExtractedEntities:
    if llm.incident_type == "Extraction Failed":
        return rules

    incident_type = (
        llm.incident_type
        if llm.incident_type not in ("Unknown", "Extraction Failed", None)
        else rules.incident_type
    )

    victims = llm.victims or rules.victims or ["Unknown"]
    suspects_raw = llm.suspect_aliases or rules.suspect_aliases or ["Unknown"]
    suspects = _clean_suspects(suspects_raw)

    phone_numbers = _merge_list(rules.phone_numbers, llm.phone_numbers)
    email_addresses = _merge_list(rules.email_addresses, llm.email_addresses)
    upi_ids = _merge_list(rules.upi_ids, llm.upi_ids)
    fraudulent_urls = _merge_list(rules.fraudulent_urls, llm.fraudulent_urls)
    ip_addresses = _merge_list(rules.ip_addresses, llm.ip_addresses)

    bank_accounts = rules.bank_accounts if rules.bank_accounts else llm.bank_accounts

    amount_lost = rules.amount_lost
    if not amount_lost or amount_lost == "N/A":
        amount_lost = llm.amount_lost or "N/A"

    crypto_wallets = (
        sorted(set(rules.cryptocurrency_wallets + llm.cryptocurrency_wallets))
        if (rules.cryptocurrency_wallets or llm.cryptocurrency_wallets)
        else ["N/A"]
    )

    return ExtractedEntities(
        case_number="N/A",
        incident_type=incident_type,
        victims=victims,
        suspect_aliases=suspects,
        investigating_officers=[],
        phone_numbers=phone_numbers,
        email_addresses=email_addresses,
        upi_ids=upi_ids,
        bank_accounts=bank_accounts,
        ip_addresses=ip_addresses,
        fraudulent_urls=fraudulent_urls,
        cryptocurrency_wallets=crypto_wallets,
        amount_lost=amount_lost,
    )


def hybrid_extract_cybercrime_entities(text: str, evidence_type: str = "generic") -> ExtractedEntities:
    rules_entities = extract_with_rules(text, evidence_type)
    llm_entities = extract_with_gemini(text, evidence_type)
    return _merge_entities(rules_entities, llm_entities)


# 6. PUBLIC ENTRY POINTS

# keep False for speed; set True if you want hybrid
USE_GEMINI_FOR_EXTRACTION = False


def extract_cybercrime_entities(text: str, evidence_type: str = "generic") -> ExtractedEntities:
    if not USE_GEMINI_FOR_EXTRACTION:
        return extract_with_rules(text, evidence_type)
    return hybrid_extract_cybercrime_entities(text, evidence_type)


def summarize_case(text: str, entities: ExtractedEntities) -> str:
    """
    Short summary (3–5 sentences) grounded in entities + evidence text.
    """
    if gemini_model is None:
        return ""

    ctx = entities.dict()

    prompt = f"""
    You are a cybercrime investigator assistant.

    Here are structured entities extracted from an evidence file:
    {json.dumps(ctx, indent=2)}

    And here is a snippet of the original evidence text:
    \"\"\"{text[:1500]}\"\"\"

    Write a concise summary in 3–5 sentences that clearly explains:
    - who is the likely victim,
    - who is the likely fraudster (names/aliases/companies),
    - what type of fraud this appears to be,
    - how much money is involved (if known),
    - which payment methods/accounts/UPIs were used.

    Be factual and cautious. Do NOT invent details that are not supported
    by the provided entities or text.
    """

    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config={"temperature": 0.3},
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            },
        )
        return (response.text or "").strip()
    except Exception as e:
        print(f"[CaseFusion] ❌ Gemini summary error: {e}")
        return ""
