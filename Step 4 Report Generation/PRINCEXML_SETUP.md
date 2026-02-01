# PrinceXML Implementation Guide

## Overview

This implementation follows ChatGPT's recommendation for "top-tier sell-side PDF" quality by:
1. Using **structured JSON output** from LLMs (instead of markdown tables)
2. Rendering with **PrinceXML** (commercial print-quality PDF engine)
3. Using **HTML/CSS templates** (Jinja2) for deterministic layout

## Installation

### 1. Install PrinceXML

Download and install from: https://www.princexml.com/download/

**macOS:**
```bash
brew install --cask prince
```

**Linux/Windows:**
Download installer from the website.

**Verify installation:**
```bash
prince --version
```

### 2. Install Python Dependencies

```bash
pip install jinja2
```

## Usage

### Compare 3 Models (Structured Output)

```bash
# Run with all 3 models using structured JSON output
python scripts/05_compare_models.py --structured

# Run with specific models
python scripts/05_compare_models.py --structured --models anthropic openai google

# Run for specific date
python scripts/05_compare_models.py --structured --date 2026-01-30
```

### Output Structure

Reports are saved to `outputs/comparison/{date}/`:

```
outputs/comparison/2026-01-30/
├── anthropic_report.json    # Structured JSON data
├── anthropic_report.md     # Raw LLM response (fallback)
├── anthropic_report.pdf     # PrinceXML-generated PDF
├── openai_report.json
├── openai_report.md
├── openai_report.pdf
├── google_report.json
├── google_report.md
├── google_report.pdf
└── comparison_summary.txt    # Quick comparison stats
```

## How It Works

### 1. Structured Prompt (`prompts/daily_wrap_structured.md`)

The LLM is instructed to output JSON with this structure:

```json
{
  "executive_synthesis": {
    "single_most_important": "...",
    "key_takeaways": ["...", "..."],
    "what_to_watch": ["...", "..."]
  },
  "flash_headlines": ["...", "...", "..."],
  "sections": [
    {
      "title": "Section Title",
      "narrative": "Analysis text...",
      "tables": [
        {
          "title": "Table Title",
          "headers": ["Col1", "Col2", "Col3"],
          "rows": [["...", "...", "..."], ...],
          "column_alignments": ["left", "right", "right"],
          "column_widths": ["40%", "30%", "30%"]
        }
      ]
    }
  ]
}
```

### 2. Parser (`utils/pdf_prince/parser.py`)

Extracts JSON from LLM response (handles markdown code blocks, extra text).

### 3. HTML Template (`utils/pdf_prince/templates/report.html`)

Jinja2 template that renders structured data into HTML with:
- Cover page
- Executive synthesis callout box
- Sections with narrative + tables
- Proper table structure with headers/rows

### 4. CSS Stylesheet (`utils/pdf_prince/templates/styles.css`)

Paged media CSS with:
- Running headers/footers
- Page numbering
- Table styling (zebra stripes, proper borders)
- Typography scale
- Cover page styling

### 5. PrinceXML Converter (`utils/pdf_prince/convert.py`)

Renders HTML → PDF using PrinceXML with:
- PDF/X-1a:2003 profile (print quality)
- Proper font embedding
- Page break control
- Table header repetition across pages

## Comparison Workflow

1. **Run comparison:**
   ```bash
   python scripts/05_compare_models.py --structured
   ```

2. **Review outputs:**
   - Open PDFs side-by-side
   - Check `comparison_summary.txt` for stats
   - Review JSON files for structured data quality

3. **Choose best model:**
   - Based on narrative quality
   - Table structure correctness
   - PDF rendering quality
   - Consistency

4. **Use chosen model:**
   - Update `03_generate_daily_report.py` to use structured mode
   - Or continue using comparison script for daily reports

## Advantages Over Current System

1. **Tables render correctly** - No more markdown pipe tables breaking
2. **Consistent layout** - HTML/CSS templates ensure repeatability
3. **Print quality** - PrinceXML handles paged media properly
4. **Structured data** - Easier to parse, validate, and transform
5. **Model comparison** - Easy side-by-side evaluation

## Fallback Behavior

If PrinceXML is not installed:
- JSON files are still saved
- Markdown fallback files are saved
- PDF generation is skipped (with warning)

## Next Steps

1. **Install PrinceXML** (required for PDF generation)
2. **Run comparison** to see all 3 models side-by-side
3. **Choose best model** based on output quality
4. **Integrate** chosen model into daily workflow

## Troubleshooting

**PrinceXML not found:**
- Verify installation: `prince --version`
- Check PATH includes PrinceXML binary
- On macOS, may need to allow in System Preferences > Security

**JSON parsing errors:**
- Check LLM response format
- Review `{model}_report.md` for raw output
- May need to adjust prompt if models aren't following JSON format

**PDF generation fails:**
- Check PrinceXML logs
- Verify HTML template is valid
- Check CSS syntax
