import os
import fitz
import easyocr
import json
from groq import Groq


#############################################
# 1. OCR Engine (EasyOCR – Windows Friendly)
#############################################

reader = easyocr.Reader(['en'], gpu=False)

import re
import json

def safe_extract_json(model_output: str):
    # Extract JSON block between first '{' and last '}'
    try:
        json_block = model_output[model_output.find("{"): model_output.rfind("}") + 1]
    except:
        raise ValueError("No JSON object found in LLM response.")

    # Remove markdown fences
    json_block = json_block.replace("```json", "").replace("```", "").strip()

    # Replace single quotes with double quotes (only if needed)
    if "'" in json_block and '"' not in json_block:
        json_block = json_block.replace("'", '"')

    # Remove trailing commas
    json_block = re.sub(r",\s*}", "}", json_block)
    json_block = re.sub(r",\s*]", "]", json_block)

    try:
        return json.loads(json_block)
    except json.JSONDecodeError as e:
        print("---- RAW JSON BLOCK ----")
        print(json_block)
        raise e


def extract_text_from_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    all_text = []

    for page in doc:
        pix = page.get_pixmap(dpi=350)
        bw = fitz.Pixmap(pix, 0)  # enforce grayscale
        img_bytes = bw.tobytes("png")

        # OCR
        result = reader.readtext(img_bytes, detail=1, paragraph=True)
        text_lines = [item[1] for item in result]
        all_text.append("\n".join(text_lines))

    return "\n\n".join(all_text).strip()


#############################################
# 2. LLM – Field Extraction (Groq)
#############################################

def extract_fields_with_groq(text: str):
    client = Groq()
    prompt = """
You are an expert in cleaning noisy OCR from Indian cab receipts (Uber, Ola, Rapido).
OCR has recurring predictable errors such as:
- Currency "₹" becomes "3" or "2"
- Amounts gain a leading "3" or "2" (e.g., 3123.02 → 123.02)
- Curly braces { appear randomly
- "3210" from Rapido means "₹210"
- Extra characters appear around addresses or promotion text
- Some fields appear on separate lines
- Commas and spaces get corrupted

Always:
1. Clean the OCR patterns based on examples below.
2. Correct leading digits in amounts:
   - If a money value begins with “3” or “2” AND the next digits form a valid amount → remove the first digit.
3. Fix formatting issues: remove { } stray characters, merge split numbers.
4. Extract fields even if labels are out of order or noisy.
5. Auto-detect INR even when symbol is missing.
6. Return ONLY valid JSON.

=====================================================
FEW-SHOT EXAMPLES
=====================================================

### EXAMPLE 1 — Uber noisy OCR
OCR INPUT:
Total
3123.02
Suggested fare
2126.82
Subtotal
{126.82
Promotion
33.80
License Plate: KA4OB1670
2.37 PM
3.04 PM
5, Nallurhalli, Whitefield
XM2X+R4W, Marathahalli

EXPECTED JSON OUTPUT:
{
  "provider": "Uber",
  "ride_id": null,
  "date": "2025-10-09",
  "start_time": "2:37 PM",
  "end_time": "3:04 PM",
  "total_amount": "123.02",
  "currency": "INR",
  "pickup_address": "5, Nallurhalli, Whitefield, Bengaluru",
  "dropoff_address": "XM2X+R4W, Marathahalli, Bengaluru",
  "vehicle_number": "KA40B1670"
}

### EXAMPLE 2 — Rapido OCR with merged number
OCR INPUT:
Selected Price 3210
Booking History rapido Ashwini RD17506969273550314
Driver name prasad
Vehicle Number KA07B5736
Jun 23rd 2025,10.17 PM
Gate 1 Forum Shantiniketan Mall
96 Tulsi Theater Rd Marathahalli

EXPECTED JSON OUTPUT:
{
  "provider": "Rapido",
  "ride_id": "RD17506969273550314",
  "date": "2025-06-23",
  "time": "10:17 PM",
  "total_amount": "210",
  "currency": "INR",
  "pickup_address": "Gate 1, Forum Shantiniketan Mall, Whitefield, Bengaluru",
  "dropoff_address": "96, Tulsi Theater Rd, Marathahalli, Bengaluru",
  "vehicle_number": "KA07B5736"
}

### EXAMPLE 3 — Rapido OCR with distorted address + spaces
OCR INPUT:
Selected Price 3210
rapido Ashwini
Ride ID RD17506969273550314
Vehicle Number KA07B5736
Jun 23rd 2025, 7:30 PM
SSC Block  Tesco HSC Whitefield 0 560066
96 Tulsi Theater Rd Marathahalli 560037

EXPECTED JSON OUTPUT:
{
  "provider": "Rapido",
  "ride_id": "RD17506969273550314",
  "date": "2025-06-23",
  "time": "7:30 PM",
  "total_amount": "210",
  "currency": "INR",
  "pickup_address": "SSC Block, Tesco HSC, Whitefield, Bengaluru 560066",
  "dropoff_address": "96, Tulsi Theater Rd, Marathahalli, Bengaluru 560037",
  "vehicle_number": "KA07B5736"
}

=====================================================
You MUST ALWAYS return ONLY valid JSON. 
No markdown. No explanation. No comments. No code. No placeholders. 
If information is missing, return null. 
If fields cannot be extracted, return null.

VALID JSON SCHEMA:
{
  "provider": string or null,
  "ride_id": string or null,
  "date": string or null,
  "start_time": string or null,
  "end_time": string or null,
  "time": string or null,
  "total_amount": string or null,
  "currency": "INR",
  "pickup_address": string or null,
  "dropoff_address": string or null,
  "vehicle_number": string or null
}

NEVER return anything except one JSON object.

Now extract fields for the following OCR text and return ONLY the JSON.
{text}
---------------------
"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    import re

    output = resp.choices[0].message.content

    # Extract strict JSON object
    matches = re.findall(r"\{.*?\}", output, flags=re.DOTALL)
    if not matches:
        raise ValueError("No JSON found in model output")

    json_str = matches[0]

    return safe_extract_json(json_str)


#############################################
# 3. Process a single PDF
#############################################

def process_single_pdf(pdf_path: str):
    text = extract_text_from_pdf(pdf_path)
    if not text or len(text) < 5:
        return {"file": pdf_path, "error": "Empty OCR output"}

    fields = extract_fields_with_groq(text)

    return {
        "file": pdf_path,
        "ocr_text": text,
        "extracted_fields": fields
    }


#############################################
# 4. Process an entire folder
#############################################

def process_folder(folder_path: str):
    if not os.path.isdir(folder_path):
        raise ValueError(f"Not a folder: {folder_path}")

    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"📄 Processing: {pdf_path}")
            result = process_single_pdf(pdf_path)
            results.append(result)

    return results


#############################################
# 5. CLI Entry Point
#############################################

if __name__ == "__main__":

    folder_path = "D:/pycharm/admin_billdesk/resources"
    output = process_folder(folder_path)

    # Specify the file path where you want to save the JSON data
    file_path = "output.json"

    # Open the file in write mode ('w') and use json.dump()
    with open(file_path, 'w') as json_file:
        json.dump(output, json_file, indent=4)  # indent for pretty-printing
