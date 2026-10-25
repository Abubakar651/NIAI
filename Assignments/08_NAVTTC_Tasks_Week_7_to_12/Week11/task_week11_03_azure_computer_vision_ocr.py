# Abubakar Iqbal
# NAVTTC Week 11 - Task 03: Azure Vision & OCR Text Extraction

import re

class AzureVisionClient:
    def analyze(self, filename):
        fname = filename.lower()
        if "salary" in fname or "slip" in fname:
            tags = ["document", "financial", "salary slip", "payroll"]
            caption = "Employee monthly salary slip with breakdown"
        elif "id" in fname or "card" in fname:
            tags = ["identity card", "national id", "photo id"]
            caption = "National identity card document"
        else:
            tags = ["document", "paper", "text"]
            caption = "Scanned document"
        return {"caption": caption, "tags": tags}

    def ocr_read(self, text):
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        out_lines = []
        for idx, line in enumerate(lines, 1):
            out_lines.append({
                "line": idx,
                "text": line,
                "bbox": [50, 40 * idx, 450, 40 * idx + 30]
            })
        return out_lines

if __name__ == "__main__":
    print("NAVTTC Week 11 Task 03 - Abubakar Iqbal")

    abubakar_client = AzureVisionClient()

    image_meta = "abubakar_salary_slip_oct2026.png"
    analysis = abubakar_client.analyze(image_meta)
    print(f"\nImage Analysis for '{image_meta}':")
    print(f"Caption: {analysis['caption']}")
    print(f"Tags:    {', '.join(analysis['tags'])}")

    abubakar_document = """
    COMPANY: NETSOL TECHNOLOGIES
    EMPLOYEE: ABUBAKAR IQBAL
    ROLE: MACHINE LEARNING ENGINEER
    BASIC PAY: PKR 140,000.00
    ALLOWANCES: PKR 35,000.00
    TAX DEDUCTION: PKR 12,500.00
    NET SALARY: PKR 162,500.00
    STATUS: PAID VIA BANK TRANSFER
    """

    print("\nAzure Read OCR Results:")
    extracted = abubakar_client.ocr_read(abubakar_document)
    for item in extracted:
        print(f" Line {item['line']}: {item['text']}")

    print("\nField Parsing from OCR Output:")
    emp_match = re.search(r"EMPLOYEE:\s*([A-Z\s]+)", abubakar_document)
    net_match = re.search(r"NET SALARY:\s*(PKR\s*[\d,]+\.\d{2})", abubakar_document)

    if emp_match:
        print(f"Employee Name: {emp_match.group(1).strip()}")
    if net_match:
        print(f"Verified Net:  {net_match.group(1).strip()}")
