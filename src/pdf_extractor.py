import pdfplumber
import re
import pandas as pd
import os
from dateutil import parser

def normalize_date(date_str):
    """Try to normalize various date formats into ISO format."""
    if not date_str:
        return None
    try:
        # dateutil can handle most cases like "Apr 5th 2025, 12:27 PM", "October 6, 2025", "10/6/25 10:13 AM"
        dt = parser.parse(date_str, fuzzy=True, dayfirst=False)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

def extract_invoice_details(pdf_path):
    text_content = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text_content += page.extract_text() + "\n"

    text = text_content.replace('\n', ' ').strip()

    # --- Regex patterns ---
    id_pattern = re.search(r'\b(RD\d{10,}|Trip ID[:\s]*[A-Z0-9\-]+|Invoice No[:\s]*[A-Z0-9\-]+)\b', text)
    date_pattern = re.search(
        r'((?:\d{1,2}/\d{1,2}/\d{2,4}\s*\d{1,2}:\d{2}\s*[APM]{2})|'
        r'(?:\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{1,2}(?:st|nd|rd|th)?\s*,?\s*\d{4}(?:,\s*\d{1,2}:\d{2}\s*[APM]{2})?)|'
        r'(?:\b[A-Za-z]+\s+\d{1,2},\s*\d{4}))',
        text
    )
    amount_pattern = re.search(r'(₹|Rs\.|\$)\s*([\d,]+\.?\d*)', text)

    # --- Extracted values ---
    bill_id = id_pattern.group(0).replace("Trip ID", "").replace("Invoice No", "").strip() if id_pattern else None
    bill_date_raw = date_pattern.group(1).strip() if date_pattern else None
    bill_date = normalize_date(bill_date_raw)
    bill_amount = amount_pattern.group(2).replace(",", "").strip() if amount_pattern else None

    platform = "Uber" if "Uber" in text else "Rapido" if "Rapido" in text else "Unknown"

    return {
        "platform": platform,
        "bill_id": bill_id,
        "bill_date": bill_date,
        "bill_amount": bill_amount,
    }

def extract_from_folder(folder_path):
    results = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            results.append(extract_invoice_details(pdf_path))
    return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    pdf_files = ["D:/pycharm/admin_billdesk/resources/CAB_RECEIP.pdf","D:/pycharm/admin_billdesk/resources/receipt_9ed71f8b-428d-4d31-8743-c3dda2c2e686.pdf","D:/pycharm/admin_billdesk/resources/CAB_RECEIPT_RD17506969273550314.pdf"]
    extracted_data = [extract_invoice_details(f) for f in pdf_files]
    df = pd.DataFrame(extracted_data)

    print(df[["platform", "bill_id", "bill_date", "bill_amount"]])
    df.to_csv("ride_invoices.csv", index=False)
