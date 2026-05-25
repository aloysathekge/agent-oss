---
name: pdf_generator
description: Generate PDF documents such as reports, summaries, and structured data exports.
triggers: pdf, generate pdf, create pdf, report, summary, export, document, write a report
---

# PDF Generator Skill

## When to use
- The user asks to generate, create, or export a PDF document.
- The user asks for a report, summary, or structured document to be saved as a PDF.
- The user asks to export data or information into a downloadable/shareable format.

## When NOT to use
- The user is discussing PDFs conceptually (how PDFs work, PDF specifications, etc.).
- The user wants to read or parse an existing PDF — this skill only creates new PDFs.
- The user mentions "report" or "summary" but wants it displayed in chat, not saved as a file.

## Tools available
- `generate_pdf(title, content, filename="output")` — creates a simple PDF with a title and body text. Good for memos, notes, and simple documents.
- `generate_report_pdf(title, sections_json, filename="report")` — creates a structured report with multiple headed sections. `sections_json` is a JSON string: `[{"heading": "...", "body": "..."}]`.
- `generate_table_pdf(title, headers_json, rows_json, filename="table")` — creates a PDF with a formatted data table. `headers_json` = JSON array of column names, `rows_json` = JSON array of row arrays.
- `list_generated_pdfs()` — lists all previously generated PDFs with file sizes and timestamps.

## Operating rules
- **Always confirm the content before generating.** If the user's request is vague ("make me a report"), ask what should be in it.
- **Use `generate_report_pdf` for multi-section documents**, `generate_pdf` for simple single-body documents, and `generate_table_pdf` when the user wants tabular data.
- **Use descriptive filenames** derived from the title or content — not generic names like "output" or "document".
- **Respect procedural memory** when composing content (tone, formatting rules, style preferences).
- Generated PDFs are uploaded to cloud storage. The tool returns a **signed download URL valid for 7 days**. Always inform the user that the link expires in 7 days.

## Examples

**User:** "Generate a PDF summary of our Q3 infrastructure changes."
→ `generate_report_pdf(title="Q3 Infrastructure Changes Summary", sections_json='[{"heading": "Overview", "body": "..."}, {"heading": "Key Changes", "body": "..."}]', filename="q3-infrastructure-summary")`

**User:** "Create a PDF with a table of our team members and their roles."
→ `generate_table_pdf(title="Team Directory", headers_json='["Name", "Role", "Email"]', rows_json='[["Elias", "Lead Data Engineer", "elias@quantumcorp.com"]]', filename="team-directory")`

**User:** "Make me a quick PDF note about the Redis outage."
→ `generate_pdf(title="Redis Outage Note", content="On ..., a memory leak in the Redis cache eviction policy caused...", filename="redis-outage-note")`

**User:** "What PDFs have I generated?"
→ `list_generated_pdfs()` → list files with sizes and dates.
