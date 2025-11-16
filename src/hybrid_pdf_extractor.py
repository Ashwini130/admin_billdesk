import os
import re
import pdfplumber
import easyocr
from pdf2image import convert_from_path
from dateutil import parser
from PIL import Image
import pandas as pd

# Create OCR reader once
reader = easyocr.Reader(["en"])


# ---------------------------
# DATE NORMALIZATION
# ---------------------------
def normalize_date(date_str):
    if not date_str:
        return None
    try:
        dt = parser.parse(date_str, fuzzy=True)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return None


# ---------------------------
# TEXT EXTRACTION (PDF + IMAGE)
# ---------------------------
def extract_text_from_pdf(pdf_path):
    text = ""

    # 1. Try text-based extraction (pdfplumber)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

    # If text extracted successfully → use it
    if len(text.strip()) > 40:   # threshold
        return text

    # 2. Otherwise fallback to OCR
    print(f"[OCR FALLBACK] Using OCR for PDF: {os.path.basename(pdf_path)}")

    images = convert_from_path(pdf_path)
    ocr_text = ""

    for img in images:
        ocr_text += " ".join(reader.readtext(img, detail=0)) + " "

    return ocr_text


def extract_text_from_image(image_path):
    img = Image.open(image_path)
    return " ".join(reader.readtext(img, detail=0))


# ---------------------------
# MAIN EXTRACTION PIPELINE
# ---------------------------
def extract_invoice(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    # Detect file type
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
        text = extract_text_from_image(file_path)
    else:
        raise ValueError("Unsupported file type: " + ext)

    # ---------------------------
    # REGEX EXTRACTION
    # ---------------------------

    # Bill or Trip ID (Rapido, Uber, Ola etc.)
    bill_id_pattern = re.search(
        r'(RD\d{10,}|[A-Z0-9]{10,})',
        text
    )

    # Date patterns
    date_pattern = re.search(
        r'(\d{1,2}/\d{1,2}/\d{2,4}\s*\d{1,2}:\d{2}\s*[APMapm]{2}|'
        r'[A-Za-z]+\s+\d{1,2},\s*\d{4}|'
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?\s+\d{4}(?:,\s*\d{1,2}:\d{2}\s*[APMapm]{2})?)',
        text
    )

    # Amount patterns
    amount_pattern = re.search(r'(₹|Rs\.|\$)\s*([\d,]+\.?\d*)', text)

    return {
        "bill_id": bill_id_pattern.group(1) if bill_id_pattern else None,
        "bill_amount": amount_pattern.group(2).replace(",", "") if amount_pattern else None,
        "bill_date_raw": date_pattern.group(1) if date_pattern else None,
        "bill_date": normalize_date(date_pattern.group(1)) if date_pattern else None,
        "text_preview": text[:300]
    }


# ---------------------------
# FOLDER BATCH PROCESSING
# ---------------------------
def extract_from_folder(folder_path):
    results = []
    for file in os.listdir(folder_path):
        if file.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".webp")):
            results.append(extract_invoice(os.path.join(folder_path, file)))
    return results

if __name__ == "__main__":
    results = extract_from_folder("D:/pycharm/admin_billdesk/resources/")
    for r in results:
        print(r)
    df = pd.DataFrame(results)
    df.to_csv("rides_hybrid_approach.csv", index=False)

