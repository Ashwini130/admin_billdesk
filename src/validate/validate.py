# validators/commute_validator.py

from dataclasses import dataclass
from typing import List, Dict, Optional

from thefuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import numpy as np


# ---------------------------------------------------
# Setup global embedding model (fast & lightweight)
# ---------------------------------------------------

MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------
# Data structures
# ---------------------------------------------------

@dataclass
class EmployeeData:
    emp_name: str
    bill_date: str
    employee_address: str
    client_addresses: List[str]


@dataclass
class ReceiptData:
    filename: str
    rider_name: Optional[str]
    date: Optional[str]
    time: Optional[str]
    pickup_address: Optional[str]
    drop_address: Optional[str]
    amount: Optional[float]
    distance_km: Optional[float]
    service_provider: Optional[str]
    ocr: Optional[str]


# ---------------------------------------------------
# Utility functions
# ---------------------------------------------------

def _embed(text: str):
    """Helper to convert string to embedding."""
    return MODEL.encode(text, convert_to_tensor=True)


def address_similarity(addr1: str, addr2: str) -> float:
    """Cosine similarity between two addresses."""
    if not addr1 or not addr2:
        return 0.0

    emb1 = _embed(addr1)
    emb2 = _embed(addr2)
    return float(util.cos_sim(emb1, emb2))


# ---------------------------------------------------
# Main Validation Logic
# ---------------------------------------------------

class CommuteValidator:

    NAME_THRESHOLD = 75       # 75% fuzzy match
    ADDRESS_THRESHOLD = 0.40  # 0.40 cosine similarity

    @staticmethod
    def validate(receipt: ReceiptData, employee: EmployeeData) -> Dict:
        """
        Apply validation rules:
        - Name fuzz match >= 75
        - Date match exact
        - Pickup/drop must match employee_address OR any client_address
          with similarity >= 0.40
        """

        results = {
            "filename": receipt.filename,
            "name_match": False,
            "date_match": False,
            "pickup_match": False,
            "drop_match": False,
            "pickup_match_score": 0.0,
            "drop_match_score": 0.0,
            "name_match_score": 0
        }

        # -------------------------
        # 1. Name Validation
        # -------------------------
        if receipt.rider_name:
            score = fuzz.token_set_ratio(receipt.rider_name, employee.emp_name)
            results["name_match_score"] = score
            if score >= CommuteValidator.NAME_THRESHOLD:
                results["name_match"] = True

        # -------------------------
        # 2. Date Validation
        # -------------------------
        if receipt.date == employee.bill_date:
            results["date_match"] = True

        # -------------------------
        # 3. Address Validation
        # -------------------------
        all_addr = [employee.employee_address] + employee.client_addresses

        # Pickup
        if receipt.pickup_address:
            scores = [address_similarity(receipt.pickup_address, a) for a in all_addr]
            best = max(scores) if scores else 0
            results["pickup_match_score"] = best
            if best >= CommuteValidator.ADDRESS_THRESHOLD:
                results["pickup_match"] = True

        # Drop
        if receipt.drop_address:
            scores = [address_similarity(receipt.drop_address, a) for a in all_addr]
            best = max(scores) if scores else 0
            results["drop_match_score"] = best
            if best >= CommuteValidator.ADDRESS_THRESHOLD:
                results["drop_match"] = True

        return results
