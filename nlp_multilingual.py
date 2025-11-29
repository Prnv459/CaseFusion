import os
import re
from typing import List, Set

from transformers import pipeline
import google.generativeai as genai

from .schemas import ExtractedEntities, BankDetails
from .nlp import extract_cybercrime_entities  # reuse your main engine

# ============================================
# 0. GOOGLE / GEMINI CONFIG (for translation)
# ============================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    translation_model = genai.GenerativeModel("gemini-2.5-flash")
else:
    translation_model = None
    print(
        "[CaseFusion] WARNING: GOOGLE_API_KEY not set. "
        "Multilingual translation will be skipped."
    )

# ============================================
# 1. MULTILINGUAL NER MODEL (FOR EXTRA NAMES)
# ============================================

print("Loading multilingual NER model (Davlan/xlm-roberta-base-ner-hRL)...")
ner_pipeline = pipeline(
    "ner",
    model="Davlan/xlm-roberta-base-ner-hrl",
    aggregation_strategy="simple",
    device=-1,  # Force CPU
)
print("Multilingual NER model loaded successfully.")

# Tokens we never want as suspects in audio (cities, payment systems, etc.)
AUDIO_SUSPECT_STOPWORDS = {
    "upi",
    "ybl",
    "gpay",
    "phonepe",
    "paytm",
    "noida",
    "delhi",
    "gurgaon",
    "mumbai",
    "bangalore",
    "bengaluru",
    "india",
}


# ============================================
# 2. TEXT NORMALIZATION (FOR AUDIO TRANSCRIPTS)
# ============================================

def normalize_transcription(text: str) -> str:
    """
    Converts spoken words (from Whisper) into symbols for regex.
    Example:
      - "abc at gmail dot com" -> "abc@gmail.com"
      - "registration phi at ybl" -> "registration.phi@ybl"
      - "double nine" -> "99"
    """
    text = f" {text} "  # padding

    # Part 1: common symbol words
    # We'll keep spaces here, then collapse later.
    text = text.replace(" at ", " @ ")
    text = text.replace(" dot ", " . ")

    # Part 2: "double <digit-word>" -> repeated digits
    doubles = {
        "zero": "00",
        "one": "11",
        "two": "22",
        "three": "33",
        "four": "44",
        "five": "55",
        "six": "66",
        "seven": "77",
        "eight": "88",
        "nine": "99",
    }
    for word, digits in doubles.items():
        text = text.replace(f" double {word} ", f" {digits} ")

    # Extra defensive replacements (if Whisper omitted "double")
    text = text.replace(" double nine ", " 99 ")
    text = text.replace(" double eight ", " 88 ")
    text = text.replace(" double seven ", " 77 ")
    text = text.replace(" double six ", " 66 ")
    text = text.replace(" double five ", " 55 ")

    # Part 3: patterns like "<token> at ybl" -> "<token>@ybl"
    # Do this BEFORE final whitespace collapse.
    # Match non-space token, then 'at' or '@', then word.
    text = re.sub(
        r"(\S+)\s+at\s+([A-Za-z0-9]+)",
        r"\1@\2",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(\S+)\s*@\s*([A-Za-z0-9]+)",
        r"\1@\2",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize whitespace
    text = " ".join(text.split())
    return text


# ============================================
# 3. TRANSLATION TO ENGLISH (OPTIONAL BUT POWERFUL)
# ============================================

def translate_to_english(text: str) -> str:
    """
    Use Gemini to translate Hinglish / Indian languages to clear English,
    while preserving names, numbers, emails, UPI handles, and URLs.
    If translation model is not available or fails, return the original text.
    """
    if translation_model is None:
        return text

    prompt = f"""
    You are a translation assistant for cybercrime investigations.

    Translate the following text into clear, grammatical English.
    VERY IMPORTANT:
    - Preserve all names, phone numbers, amounts, emails, URLs, UPI IDs, and bank details.
    - Do NOT summarize, do NOT add or remove information.
    - Output ONLY the translated text, nothing else.

    TEXT:
    \"\"\"{text}\"\"\"
    """

    try:
        response = translation_model.generate_content(
            prompt,
            generation_config={"temperature": 0.2},
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            },
        )
        if not response.text:
            return text
        return response.text.strip()
    except Exception as e:
        print(f"[CaseFusion] ❌ Multilingual translation error: {str(e)}")
        return text


# ============================================
# 4. HELPERS FOR AUGMENTATION
# ============================================

def _augment_list(current: List[str], found: List[str]) -> List[str]:
    """
    Merge new found values into a current list, filtering out 'N/A' and 'Unknown'.
    """
    clean_current: Set[str] = {
        v for v in (current or []) if v and v not in ("N/A", "Unknown")
    }
    clean_found: Set[str] = {
        v for v in (found or []) if v and v not in ("N/A", "Unknown")
    }

    merged = sorted(clean_current.union(clean_found))
    return merged or (current if current else ["N/A"])


def _infer_loss_amount_from_normalized(normalized_text: str) -> str:
    """
    Audio-specific fallback:
    Look for numeric amounts near words like 'refundable', 'security',
    'deposit', 'fee', 'pay', 'payment', 'rupees', etc.
    """
    candidates = []

    # Any sequence of 3+ digits (to avoid 10, 20 etc unless context is strong)
    for m in re.finditer(r"\b(\d{3,})\b", normalized_text):
        val_str = m.group(1).replace(",", "")
        try:
            val = float(val_str)
        except ValueError:
            continue

        # Ignore very small values (likely noise)
        if val < 100:
            continue

        start = max(0, m.start() - 80)
        end = min(len(normalized_text), m.end() + 80)
        ctx = normalized_text[start:end].lower()

        score = 1.0
        if any(w in ctx for w in ["refundable", "security", "deposit", "fee", "charge"]):
            score += 3
        if any(w in ctx for w in ["pay", "paid", "send", "sent", "transfer", "transferred"]):
            score += 2
        if any(w in ctx for w in ["salary", "payout", "pending amount", "pending"]):
            score -= 2
        if "rupee" in ctx or "rs" in ctx or "₹" in ctx:
            score += 1

        candidates.append((score, val))

    if not candidates:
        return "N/A"

    candidates.sort(reverse=True)  # highest score first
    best_val = candidates[0][1]
    return f"INR {best_val:,.2f}"


def _clean_audio_suspects(suspects: List[str]) -> List[str]:
    """
    Audio-specific cleanup for suspect_aliases:
    - Remove cities, generic tokens, payment systems like 'YBL', 'UPI'.
    """
    cleaned: List[str] = []
    for s in suspects or []:
        if not s or s in ("N/A", "Unknown"):
            continue
        low = s.strip().lower()
        if low in AUDIO_SUSPECT_STOPWORDS:
            continue
        if s not in cleaned:
            cleaned.append(s)
    return cleaned or ["N/A"]


# ============================================
# 5. MAIN MULTILINGUAL EXTRACTION FUNCTION
# ============================================

def extract_multilingual_entities(text: str) -> ExtractedEntities:
    """
    Extract entities from multilingual / Hinglish audio transcripts or text.

    Pipeline:
    1. Normalize the transcript (fix 'at', 'dot', 'double nine', etc.).
    2. Translate to English using Gemini (if available).
    3. Run the main extractor (rules or hybrid) on the English text.
    4. Run multilingual NER + regex on the ORIGINAL normalized text to
       augment phones/emails/UPIs/banks/IPs and add extra names.
    5. Audio-specific cleanup (suspects, loss amount).
    """

    # 1) Normalize spoken/text artifacts
    normalized_text = normalize_transcription(text)

    # 2) Translate to English (so your main extractor works well)
    translated_text = translate_to_english(normalized_text)

    # 3) Run your main extractor on the translated text
    #    evidence_type="audio" is an appropriate hint.
    base_entities: ExtractedEntities = extract_cybercrime_entities(
        translated_text,
        evidence_type="audio",
    )

    # We'll mutate this instance
    entities = base_entities

    # 4) Multilingual NER on ORIGINAL text to catch non-English names
    try:
        ner_results = ner_pipeline(text)
        persons = {
            ent["word"].strip()
            for ent in ner_results
            if ent.get("entity_group") == "PER"
        }
    except Exception as e:
        print(f"[CaseFusion] ❌ Multilingual NER error: {str(e)}")
        persons = set()

    # Add extra person names as suspects if they aren't already victims/suspects
    extra_suspects = [
        p
        for p in persons
        if p not in (entities.victims or [])
        and p not in (entities.suspect_aliases or [])
    ]
    if extra_suspects:
        entities.suspect_aliases = sorted(
            set((entities.suspect_aliases or []) + extra_suspects)
        )

    # 5) Regex augmentation on NORMALIZED text (for numbers/emails/upi/ip/etc.)

    # Phones: any 10-digit sequence
    phone_numbers = list(set(re.findall(r"\b(\d{10})\b", normalized_text)))
    entities.phone_numbers = _augment_list(
        entities.phone_numbers, phone_numbers)

    # Emails
    email_addresses = list(
        set(
            re.findall(
                r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
                normalized_text,
            )
        )
    )
    entities.email_addresses = _augment_list(
        entities.email_addresses, email_addresses
    )

    # UPI IDs – expanded handle list, case-insensitive
    upi_pattern = r"\b[\w.\-]+@(?:ybl|okicici|okhdfcbank|oksbi|okaxis|upi|paytm|ibl|axl|oksbi)\b"
    # UPI IDs – expanded handle list, case-insensitive
    upi_pattern = r"\b[\w.\-]+@(?:ybl|okicici|okhdfcbank|oksbi|okaxis|upi|paytm|ibl|axl|oksbi)\b"
    upi_ids = list(
        set(
            re.findall(
                upi_pattern,
                normalized_text,
                flags=re.IGNORECASE,
            )
        )
    )

    # Normalize everything to lowercase and dedupe case-insensitively
    existing_upis = [
        u.lower()
        for u in (entities.upi_ids or [])
        if u and u not in ("N/A", "Unknown")
    ]
    new_upis = [u.lower() for u in upi_ids]

    merged_upis = sorted(set(existing_upis + new_upis))
    entities.upi_ids = merged_upis or ["N/A"]

    # Bank accounts (9–18 digits that are not obvious phones)
    candidate_numbers = re.findall(r"\b\d{9,18}\b", normalized_text)
    existing_accs = {b.account_number for b in (entities.bank_accounts or [])}
    extra_banks: List[BankDetails] = []

    for num in candidate_numbers:
        if len(num) == 10 and num[0] in "6789":
            continue
        if num in existing_accs:
            continue
        extra_banks.append(BankDetails(account_number=num))

    if extra_banks:
        entities.bank_accounts = (entities.bank_accounts or []) + extra_banks

    # IP addresses
    ip_addresses = list(
        set(
            re.findall(
                r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
                normalized_text,
            )
        )
    )
    entities.ip_addresses = _augment_list(entities.ip_addresses, ip_addresses)

    # 6) Audio-specific cleanup for suspects (remove YBL/UPI/Noida/etc.)
    entities.suspect_aliases = _clean_audio_suspects(
        entities.suspect_aliases
    )

    # 7) Audio-specific fallback for amount lost
    if not entities.amount_lost or entities.amount_lost == "N/A":
        inferred = _infer_loss_amount_from_normalized(normalized_text)
        if inferred != "N/A":
            entities.amount_lost = inferred

    return entities
