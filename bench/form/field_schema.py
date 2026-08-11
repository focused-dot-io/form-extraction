"""Shared field schema for the extraction benchmark.

Every extractor (OpenAI, Gemini, Claude, OCR) targets this exact set of
fields so results are directly comparable. Field types are deliberately
mixed to stress different extraction skills: plain text, dates, a numeric
ID, an email, phone numbers, and constrained categorical choices.
"""

from pydantic import BaseModel, Field

DEPARTMENTS = ["Engineering", "Sales", "Marketing", "Operations", "Finance", "Human Resources"]
EMPLOYMENT_TYPES = ["Full-time", "Part-time", "Contract"]


class FormFields(BaseModel):
    full_name: str = Field(description="Employee's full name as printed or handwritten at the top of the form")
    date_of_birth: str = Field(description="Date of birth, formatted YYYY-MM-DD")
    employee_id: str = Field(description="6-digit employee ID number")
    email: str = Field(description="Employee email address")
    phone_number: str = Field(description="Employee phone number, digits only, e.g. 5551234567")
    department: str = Field(description=f"One of: {', '.join(DEPARTMENTS)}")
    start_date: str = Field(description="Employment start date, formatted YYYY-MM-DD")
    employment_type: str = Field(description=f"One of: {', '.join(EMPLOYMENT_TYPES)}")
    emergency_contact_name: str = Field(description="Full name of emergency contact")
    emergency_contact_phone: str = Field(description="Emergency contact phone number, digits only")
    signature_name: str = Field(description="Name written on the signature line at the bottom of the form")
    signature_date: str = Field(description="Date next to the signature, formatted YYYY-MM-DD")


FIELD_NAMES = list(FormFields.model_fields.keys())

# Fields graded by the LangSmith evaluators. `signature_name` is deliberately
# excluded: for an illegible signature, "correct" ground truth is a guess,
# not a fact, so scoring it penalizes models for not reading our minds.
# It's still extracted and shown in the comparison table — just not graded.
SCORED_FIELD_NAMES = [name for name in FIELD_NAMES if name != "signature_name"]

JSON_SCHEMA = FormFields.model_json_schema()
# Every provider's structured-output mode wants a closed schema.
JSON_SCHEMA["additionalProperties"] = False
JSON_SCHEMA["required"] = FIELD_NAMES


# Must exactly match the label strings drawn on the form in generate_form.py —
# the OCR baseline anchors on these to find each field's handwritten value.
FIELD_LABELS = {
    "full_name": "Full Name",
    "date_of_birth": "Date of Birth (MM/DD/YYYY)",
    "employee_id": "Employee ID (6 digits)",
    "email": "Email Address",
    "phone_number": "Phone Number",
    "department": "Department",
    "start_date": "Start Date (MM/DD/YYYY)",
    "employment_type": "Employment Type (check one)",
    "emergency_contact_name": "Emergency Contact Name",
    "emergency_contact_phone": "Emergency Contact Phone",
    "signature_name": "Signature",
    "signature_date": "Date (MM/DD/YYYY)",
}
