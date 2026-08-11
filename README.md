# form-extraction-bench

Pits Gemini, OpenAI, Claude, and plain Tesseract OCR against each other on
the same task: extracting 12 fields from a filled-in "Employee Onboarding
Form" — some copies typed digitally, some handwritten (neat and messy).
Built on LangGraph for the side-by-side run, and LangSmith for tracking
accuracy across experiments as you iterate.

## 1. Setup

### System dependencies

```bash
# macOS
brew install tesseract poppler
```

Tesseract powers the OCR baseline; poppler is what `pdf2image` uses to
rasterize PDF pages.

### Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### API keys

```bash
cp .env.example .env
```

Fill in `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, and
`LANGSMITH_API_KEY`. Model IDs are overridable in `.env` if you want to
swap in a cheaper or newer model per provider later.

## 2. Generate and fill the form

```bash
python -m bench.form.generate_form
```

Writes `data/blank/employee_form.pdf`. Print a handful of copies:

- Fill some **by hand** — mix neat handwriting and deliberately messy
  handwriting, in pen and pencil, across a few copies.
- Fill others **digitally** (open the PDF in Preview/Acrobat, type into
  the lines with the annotation tool, export).

Photograph the handwritten ones (phone camera is fine — natural lighting,
avoid heavy shadows on the checkboxes) and save the digital ones as
PDF or PNG. Drop everything into `data/filled/`, e.g.:

```
data/filled/
  handwritten_neat_01.jpg
  handwritten_messy_01.jpg
  digital_01.pdf
```

## 3. Write ground truth

For each file in `data/filled/`, create a JSON file with the same
basename in `data/ground_truth/` containing the true values for all 12
fields (see `bench/form/field_schema.py` for the exact field names):

```json
{
  "full_name": "Jane Doe",
  "date_of_birth": "1990-04-12",
  "employee_id": "104233",
  "email": "jane.doe@example.com",
  "phone_number": "5551234567",
  "department": "Engineering",
  "start_date": "2026-09-01",
  "employment_type": "Full-time",
  "emergency_contact_name": "John Doe",
  "emergency_contact_phone": "5559876543",
  "signature_name": "Jane Doe",
  "signature_date": "2026-08-11"
}
```

Dates can be written in any recognizable format in ground truth — the
evaluator normalizes both sides before comparing. Phone/ID fields are
compared as digits-only.

## 4. Quick comparison on one form

Runs all four extractors in parallel via LangGraph and prints a table:

```bash
python run.py data/filled/handwritten_messy_01.jpg
```

## 5. Full evaluation across your form set

Upload the dataset once (re-run any time you add more filled forms):

```bash
python -m bench.eval.upload_dataset
```

Run an experiment per provider — each shows up as its own experiment in
LangSmith, so you can compare them side by side in the UI:

```bash
python -m bench.eval.run_experiment --provider all
# or just one:
python -m bench.eval.run_experiment --provider claude
```

Each experiment is scored on two metrics:

- **field_accuracy** — fraction of the 12 fields correct (partial credit)
- **exact_match** — 1.0 only if every field is correct

Open the LangSmith project named in `LANGSMITH_PROJECT` to compare the
four experiments (`extraction-openai-*`, `extraction-gemini-*`,
`extraction-claude-*`, `extraction-ocr-*`) — sort by `field_accuracy` to
see which provider (and which form variants — typed vs. neat handwriting
vs. messy handwriting) each model struggles with.

## Project layout

```
bench/
  form/field_schema.py       shared 12-field schema (source of truth)
  form/generate_form.py      generates the blank PDF form
  extraction/image_utils.py  normalizes PDF/photo input to PNG bytes
  extraction/extractors/     one module per method: openai, gemini, claude, ocr
  extraction/graph.py        LangGraph fan-out/fan-in across all four
  eval/evaluators.py         field-level scoring for LangSmith
  eval/upload_dataset.py     builds the LangSmith dataset from data/
  eval/run_experiment.py     runs one LangSmith experiment per provider
run.py                       CLI: compare all four on one image
data/
  blank/                     generated blank form
  filled/                    your photographed/exported filled forms
  ground_truth/              hand-written JSON answer keys, one per filled form
```

## Notes on the OCR baseline

`ocr_extractor` is deliberately weak: it runs Tesseract, then anchors on
the printed label text to guess which line holds each value. It can't
read which checkbox is checked at all, so `employment_type` always comes
back blank from OCR — that's a real limitation of the approach, not a
bug, and a useful data point when comparing against the vision models.
