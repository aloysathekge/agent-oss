"""LangChain @tool functions for the PDF Generator skill.

Each function creates a PDF document using fpdf2, uploads it to Google Cloud
Storage, and returns a signed download URL valid for 7 days. The LLM decides
when to call these based on the skill.md manifest.
"""
import json
import os
from datetime import datetime, timedelta

from fpdf import FPDF
from google.cloud import storage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
GCS_PROJECT_ID = os.environ.get("GCS_PROJECT_ID")


def _get_bucket():
    """Return the GCS bucket object."""
    if not GCS_BUCKET_NAME:
        raise ValueError("GCS_BUCKET_NAME not set in environment.")
    client = storage.Client(project=GCS_PROJECT_ID) if GCS_PROJECT_ID else storage.Client()
    return client.bucket(GCS_BUCKET_NAME)


def _safe_filename(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_ " else "" for c in name)
    return safe.strip().replace(" ", "-") or "output"


def _upload_to_gcs(pdf_bytes: bytes, filename: str) -> str:
    """Upload PDF bytes to GCS and return a 7-day signed URL."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    blob_name = f"{timestamp}-{filename}.pdf"

    bucket = _get_bucket()
    blob = bucket.blob(blob_name)
    blob.upload_from_string(bytes(pdf_bytes), content_type="application/pdf")

    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(days=7),
        method="GET",
    )

    return url


@tool
def generate_pdf(title: str, content: str, config: RunnableConfig, filename: str = "output") -> str:
    """Create a simple PDF with a title and body text. Returns a download link."""
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id: return "Error: Could not authenticate user identity."

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    pdf.multi_cell(0, 6, content)

    safe_name = _safe_filename(filename)
    pdf_bytes = pdf.output()
    url = _upload_to_gcs(pdf_bytes, safe_name)

    return f"PDF generated: {url}\n\nNote: This link expires in 7 days."


@tool
def generate_report_pdf(title: str, sections_json: str, config: RunnableConfig, filename: str = "report") -> str:
    """Create a structured PDF report with multiple sections.

    Args:
        title: Report title displayed at the top.
        sections_json: JSON string — array of objects with "heading" and "body" keys.
            Example: [{"heading": "Overview", "body": "Text here..."}, ...]
        filename: Output filename (without .pdf extension).
    """
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id: return "Error: Could not authenticate user identity."

    sections = json.loads(sections_json)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 14, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(8)

    for section in sections:
        heading = section.get("heading", "")
        body = section.get("body", "")

        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, body)
        pdf.ln(6)

    safe_name = _safe_filename(filename)
    pdf_bytes = pdf.output()
    url = _upload_to_gcs(pdf_bytes, safe_name)

    return f"Report PDF generated ({len(sections)} sections): {url}\n\nNote: This link expires in 7 days."


@tool
def generate_table_pdf(title: str, headers_json: str, rows_json: str, config: RunnableConfig, filename: str = "table") -> str:
    """Create a PDF containing a formatted data table.

    Args:
        title: Table title displayed at the top.
        headers_json: JSON array of column header strings. Example: '["Name", "Role"]'
        rows_json: JSON array of row arrays. Example: '[["Alice", "Engineer"], ["Bob", "Designer"]]'
        filename: Output filename (without .pdf extension).
    """
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id: return "Error: Could not authenticate user identity."

    headers = json.loads(headers_json)
    rows = json.loads(rows_json)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page("L" if len(headers) > 5 else "P")

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    page_width = pdf.w - 20
    col_width = page_width / len(headers)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(230, 230, 230)
    for header in headers:
        pdf.cell(col_width, 8, str(header), border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    for row in rows:
        for i, cell in enumerate(row):
            pdf.cell(col_width, 7, str(cell)[:50], border=1)
        pdf.ln()

    safe_name = _safe_filename(filename)
    pdf_bytes = pdf.output()
    url = _upload_to_gcs(pdf_bytes, safe_name)

    return f"Table PDF generated ({len(rows)} rows, {len(headers)} columns): {url}\n\nNote: This link expires in 7 days."


@tool
def list_generated_pdfs(config: RunnableConfig) -> str:
    """List all previously generated PDF files in cloud storage."""
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id: return "Error: Could not authenticate user identity."

    bucket = _get_bucket()
    blobs = list(bucket.list_blobs())

    if not blobs:
        return "No PDFs generated yet."

    lines = []
    for blob in sorted(blobs, key=lambda b: b.updated, reverse=True):
        size_kb = blob.size / 1024
        updated = blob.updated.strftime("%Y-%m-%d %H:%M")
        lines.append(f"{blob.name} | {size_kb:.1f} KB | {updated}")

    return "\n".join(lines)
