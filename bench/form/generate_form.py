"""Generates the blank one-page form used for the extraction benchmark.

Run this once to produce data/blank/employee_form.pdf. Print several
copies: fill some by hand (mix neat and messy handwriting) and fill
others digitally (e.g. in Preview/Acrobat, or by editing this script to
draw sample text instead of blank lines). Photograph the handwritten
ones and export the digital ones to PDF/PNG.
"""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from bench.form.field_schema import EMPLOYMENT_TYPES

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = REPO_ROOT / "data" / "blank" / "employee_form.pdf"

LINE_FIELDS = [
    ("Full Name", 6.5),
    ("Date of Birth (MM/DD/YYYY)", 5.5),
    ("Employee ID (6 digits)", 5.5),
    ("Email Address", 6.5),
    ("Phone Number", 5.5),
    ("Department", 5.5),
    ("Start Date (MM/DD/YYYY)", 5.5),
]

FOOTER_FIELDS = [
    ("Emergency Contact Name", 6.5),
    ("Emergency Contact Phone", 5.5),
]


def draw_labeled_line(c, x, y, label, line_width_in):
    c.setFont("Helvetica", 9)
    c.drawString(x, y + 14, label)
    c.setLineWidth(0.75)
    c.line(x, y, x + line_width_in * inch, y)


def generate(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    margin = 0.75 * inch
    y = height - 1 * inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Employee Onboarding Form")
    y -= 0.45 * inch

    c.setFont("Helvetica", 9)
    c.drawString(margin, y, "Please print clearly. Fields marked with a line should be filled in by hand.")
    y -= 0.5 * inch

    for label, width_in in LINE_FIELDS:
        draw_labeled_line(c, margin, y, label, width_in)
        y -= 0.55 * inch

    # Employment type checkbox group.
    c.setFont("Helvetica", 9)
    c.drawString(margin, y + 14, "Employment Type (check one)")
    box_x = margin
    for option in EMPLOYMENT_TYPES:
        c.rect(box_x, y - 2, 10, 10, stroke=1, fill=0)
        c.drawString(box_x + 14, y, option)
        box_x += 1.7 * inch
    y -= 0.6 * inch

    for label, width_in in FOOTER_FIELDS:
        draw_labeled_line(c, margin, y, label, width_in)
        y -= 0.55 * inch

    y -= 0.3 * inch
    c.setLineWidth(0.75)
    c.line(margin, y, margin + 3.5 * inch, y)
    c.setFont("Helvetica", 9)
    c.drawString(margin, y - 14, "Signature")
    c.line(margin + 4 * inch, y, margin + 6 * inch, y)
    c.drawString(margin + 4 * inch, y - 14, "Date (MM/DD/YYYY)")

    c.showPage()
    c.save()
    return output_path


if __name__ == "__main__":
    path = generate()
    print(f"Wrote blank form to {path}")
