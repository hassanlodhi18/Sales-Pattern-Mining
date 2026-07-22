import pdfplumber
import pandas as pd
import re
from pathlib import Path

pdf_path = Path("sample.pdf")
output_file = Path("psdp_final.xlsx")

rows = []

with pdfplumber.open(pdf_path) as pdf:

    for page_no, page in enumerate(pdf.pages):

        text = page.extract_text()
        if not text:
            continue

        lines = text.split("\n")

        for line in lines:

            line = line.strip()

            # sirf project rows (number se start)
            if re.match(r'^\d+\s+', line):

                parts = re.split(r'\s{2,}', line)

                rows.append(parts)

        print(f"Page {page_no+1} processed")


# =========================
# Make DataFrame
# =========================
df = pd.DataFrame(rows)


# =========================
# Fix unequal columns
# =========================
max_cols = df.shape[1]

column_names = [
    "Sr No",
    "Project Name",
    "Location/Province",
    "Total Cost",
    "Foreign Aid",
    "Local Cost",
    "Allocation",
    "Expenditure",
]

# jitne columns hain utne hi names lo
df.columns = column_names[:max_cols]


# =========================
# Clean spaces
# =========================
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)


# =========================
# Save Excel
# =========================
print("Saving at:", output_file.resolve())

df.to_excel(output_file, index=False)

print("✅ DONE → psdp_final.xlsx created")
