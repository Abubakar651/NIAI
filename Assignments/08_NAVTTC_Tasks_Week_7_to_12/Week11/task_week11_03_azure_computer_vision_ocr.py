# Week 11 - Task 03: Azure Computer Vision & OCR Text Extraction Solution
# NAVTTC Outline Week 11 Days 4 & 5
# Reference: Microsoft Learn / Analyze Images with Azure AI Vision & Extract Text

import json
import re

# Student: Abubakar Iqbal | NAVTTC / NIAI AI & Deep Learning

class AzureComputerVisionClient:
    """
    Simulated & Live-Ready Client for Azure AI Vision Service (v4.0 API).
    Capabilities:
      1. Image Tagging and Dense Captions
      2. Visual Object Detection with Bounding Boxes
      3. Read API / OCR (Extracting printed and handwritten text from documents/receipts)
    """
    def __init__(self, endpoint="https://eastus.api.cognitive.microsoft.com/", api_key="AZURE_AI_KEY_MOCK"):
        self.endpoint = endpoint
        self.api_key = api_key

    def analyze_image(self, image_metadata):
        """Simulates Azure AI Vision 'analyze' endpoint returning tags and caption."""
        desc = image_metadata.get("filename", "").lower()
        tags = []
        caption = "An uploaded financial or business document"
        confidence = 0.94

        if "salary" in desc or "slip" in desc or "payroll" in desc:
            tags = ["document", "paper", "text", "financial", "salary slip", "earnings statement"]
            caption = "A scanned employee monthly salary slip showing earnings and deductions"
        elif "id" in desc or "card" in desc or "passport" in desc:
            tags = ["identity document", "card", "text", "photo id", "national identity"]
            caption = "A national identity card showing personal photo and credentials"
        elif "receipt" in desc or "bill" in desc:
            tags = ["receipt", "text", "retail", "transaction", "amount"]
            caption = "A store transaction receipt with itemized charges and total amount"

        return {
            "caption": {"text": caption, "confidence": confidence},
            "tags": [{"name": t, "confidence": round(0.85 + (i * 0.02), 2)} for i, t in enumerate(tags)],
            "metadata": {"width": 1920, "height": 1080, "format": "Png"}
        }

    def extract_text_ocr(self, raw_document_text):
        """
        Simulates the Azure AI Vision 'Read' OCR API.
        Extracts structured blocks, text lines, and bounding boxes.
        """
        lines = [line.strip() for line in raw_document_text.strip().split("\n") if line.strip()]
        ocr_lines = []

        for idx, line_str in enumerate(lines, 1):
            words = line_str.split()
            # Bounding box polygon: [x0, y0, x1, y1, x2, y2, x3, y3]
            bbox = [50, 40 * idx, 450, 40 * idx, 450, 40 * idx + 30, 50, 40 * idx + 30]
            ocr_lines.append({
                "lineNumber": idx,
                "text": line_str,
                "boundingBox": bbox,
                "words": [{"text": w, "confidence": 0.98} for w in words]
            })

        return {
            "status": "succeeded",
            "readResults": [
                {
                    "page": 1,
                    "angle": 0.0,
                    "width": 800,
                    "height": 1100,
                    "unit": "pixel",
                    "lines": ocr_lines
                }
            ]
        }


if __name__ == "__main__":
    print("=" * 65)
    print("Week 11 Task 3: Azure AI Vision Analysis & OCR Implementation")
    print("Student: Abubakar Iqbal | NAVTTC Week 11")
    print("=" * 65)

    client = AzureComputerVisionClient()

    # 1. Image Analysis & Tagging
    image_meta = {"filename": "applicant_salary_slip_2026.png"}
    analysis_res = client.analyze_image(image_meta)

    print(f"\n[1] Azure AI Vision Tagging for '{image_meta['filename']}':")
    print(f"  Generated Caption: \"{analysis_res['caption']['text']}\" (Confidence: {analysis_res['caption']['confidence']})")
    print("  Identified Vision Tags:")
    for tag in analysis_res["tags"]:
        print(f"    - {tag['name']:<25} (Confidence: {tag['confidence']})")

    # 2. Azure AI Read API (OCR on Banking Document)
    sample_salary_slip = """
    EMPLOYER: NETSOL TECHNOLOGIES LTD
    EMPLOYEE NAME: ABUBAKAR IQBAL
    DESIGNATION: MACHINE LEARNING ANALYST
    BASIC SALARY: PKR 140,000.00
    ALLOWANCES: PKR 35,000.00
    TAX DEDUCTION: PKR 12,500.00
    NET PAYABLE: PKR 162,500.00
    STATUS: VERIFIED & TRANSFERRED
    """

    print("\n[2] Executing Azure Read API (OCR Document Extraction):")
    ocr_res = client.extract_text_ocr(sample_salary_slip)
    extracted_lines = ocr_res["readResults"][0]["lines"]

    print(f"  OCR Status: {ocr_res['status'].upper()} | {len(extracted_lines)} Text Lines Extracted:")
    for item in extracted_lines:
        print(f"  Line {item['lineNumber']:2d} [BBox {item['boundingBox'][:2]}]: {item['text']}")

    # Key-Value Pair Extraction regex demonstration
    print("\n[3] Automated Information Extraction from OCR Text:")
    net_match = re.search(r"NET PAYABLE:\s*(PKR\s*[\d,]+\.\d{2})", sample_salary_slip)
    emp_match = re.search(r"EMPLOYEE NAME:\s*([A-Z\s]+)", sample_salary_slip)
    if emp_match:
        print(f"  Parsed Applicant Name: {emp_match.group(1).strip()}")
    if net_match:
        print(f"  Verified Net Income:   {net_match.group(1).strip()}")

    print("\nWeek 11 Task 3 Azure Vision & OCR Solution Completed.")
