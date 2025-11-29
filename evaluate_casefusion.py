import os
import json
import requests
from typing import List, Dict, Set, Tuple

BASE_URL = "http://127.0.0.1:8000/analyze_cyber_report"
EVAL_FILES_DIR = "eval_data"
GROUND_TRUTH_PATH = os.path.join("evaluation", "ground_truth.json")

# 1. Utility: metrics


def compute_pr(recall_inputs: Tuple[int, int, int]) -> Tuple[float, float, float]:
    tp, fp, fn = recall_inputs
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def normalize_str_set(values: List[str]) -> Set[str]:
    """
    Lowercase + strip + ignore 'N/A' and empty.
    """
    out = set()
    for v in values or []:
        if not v:
            continue
        v2 = v.strip()
        if not v2 or v2.upper() == "N/A":
            continue
        out.add(v2.lower())
    return out


# ---------- 2. Load ground truth ----------

with open(GROUND_TRUTH_PATH, "r") as f:
    gt_list = json.load(f)

GROUND_TRUTH: Dict[str, dict] = {item["id"]: item for item in gt_list}


# ---------- 3. Define evaluation cases ----------

TEST_CASES = [
    {
        "id": "crypto_email_case",
        "files": [
            {"name": "Case 1 mail.pdf", "mime": "application/pdf"}
        ],
    },
    {
        "id": "salary_email_case",
        "files": [
            {"name": "case mail 2.pdf", "mime": "application/pdf"}
        ],
    },
    {
        "id": "whatsapp_chat_case",
        "files": [
            {"name": "s1.jpeg", "mime": "image/jpeg"},
            {"name": "s2.jpeg", "mime": "image/jpeg"},
            {"name": "s3.jpeg", "mime": "image/jpeg"},
            {"name": "s4.jpeg", "mime": "image/jpeg"},
            {"name": "s5.jpeg", "mime": "image/jpeg"},
        ],
    },
    {
        "id": "audio_call_case",
        "files": [
            {"name": "recording 3.m4a", "mime": "audio/x-m4a"}
        ],
    },
]


# ---------- 4. Talk to your API ----------

def call_casefusion(case_def: dict) -> dict:
    """
    Sends one evaluation case (one or more files) to /analyze_cyber_report
    and returns the JSON response.
    """
    files_payload = []
    file_handles = []  # keep track so we can close them safely

    for f in case_def["files"]:
        file_path = os.path.join(EVAL_FILES_DIR, f["name"])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")

        fh = open(file_path, "rb")
        file_handles.append(fh)

        files_payload.append(
            (
                "files",
                (f["name"], fh, f["mime"]),
            )
        )

    try:
        resp = requests.post(BASE_URL, files=files_payload)
        resp.raise_for_status()
        return resp.json()
    finally:
        # ALWAYS close files, even if request fails
        for fh in file_handles:
            fh.close()


# ---------- 5. Evaluation loop ----------

def main():
    # Global counts per entity type
    stats = {
        "phone": {"tp": 0, "fp": 0, "fn": 0},
        "email": {"tp": 0, "fp": 0, "fn": 0},
        "bank": {"tp": 0, "fp": 0, "fn": 0},
    }

    print("Running CaseFusion evaluation...\n")

    for case in TEST_CASES:
        cid = case["id"]
        gt = GROUND_TRUTH[cid]
        print(f"--- Evaluating {cid}: {gt['description']} ---")

        # 1) Call API
        response = call_casefusion(case)
        entities = response["extracted_entities"]

        # 2) System predictions (normalize)
        sys_phones = normalize_str_set(entities.get("phone_numbers", []))
        sys_emails = normalize_str_set(entities.get("email_addresses", []))

        # bank: only compare account_number
        sys_bank_numbers = normalize_str_set(
            [b.get("account_number", "")
             for b in entities.get("bank_accounts", [])]
        )

        # 3) Ground truth (normalize)
        gt_phones = normalize_str_set(gt.get("phones", []))
        gt_emails = normalize_str_set(gt.get("emails", []))
        gt_banks = normalize_str_set(gt.get("bank_accounts", []))

        # 4) Compute per-case TP/FP/FN and accumulate

        # Phone
        tp = len(sys_phones & gt_phones)
        fp = len(sys_phones - gt_phones)
        fn = len(gt_phones - sys_phones)
        stats["phone"]["tp"] += tp
        stats["phone"]["fp"] += fp
        stats["phone"]["fn"] += fn
        print(f"Phone   -> TP={tp}, FP={fp}, FN={fn}")

        # Email
        tp = len(sys_emails & gt_emails)
        fp = len(sys_emails - gt_emails)
        fn = len(gt_emails - sys_emails)
        stats["email"]["tp"] += tp
        stats["email"]["fp"] += fp
        stats["email"]["fn"] += fn
        print(f"Email   -> TP={tp}, FP={fp}, FN={fn}")

        # Bank
        tp = len(sys_bank_numbers & gt_banks)
        fp = len(sys_bank_numbers - gt_banks)
        fn = len(gt_banks - sys_bank_numbers)
        stats["bank"]["tp"] += tp
        stats["bank"]["fp"] += fp
        stats["bank"]["fn"] += fn
        print(f"Bank    -> TP={tp}, FP={fp}, FN={fn}")

        print()

    # ---------- 6. Final metrics table ----------
    print("\n================ OVERALL METRICS ================\n")
    print(f"{'Entity':<10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
    print("-" * 45)

    for label in ["phone", "email", "bank"]:
        tp = stats[label]["tp"]
        fp = stats[label]["fp"]
        fn = stats[label]["fn"]
        p, r, f1 = compute_pr((tp, fp, fn))
        print(
            f"{label.capitalize():<10} "
            f"{p:>10.2f} {r:>10.2f} {f1:>10.2f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
