import fitz  # PyMuPDF
import re

# Define section keywords (not requiring exact matches)
section_patterns = {
    "Abstract": r'\babstract\b',
    "Introduction": r'\b(introduction|problem formulation|overview|preliminaries|motivation)\b',
    "Related Work": r'\b(related work|literature review)\b',
    "Method": r'\b(methodology|method|approach|system design|architecture)\b',
    "Experiment": r'\b(experiments?|evaluation|case studies|implementation|results|findings)\b',
    "Conclusion": r'\b(conclusion|summary|discussion|closing remarks)\b',
    "References": r'\b(references|bibliography)\b'
}

def extract_full_text_by_blocks(pdf_path: str, output_txt_path: str):
    """
    Extract full text in block order (friendly for double-column layout)
    """
    doc = fitz.open(pdf_path)
    with open(output_txt_path, 'w', encoding='utf-8') as out:
        for page in doc:
            blocks = page.get_text("blocks")
            text_blocks = [b for b in blocks if b[5] == 0]
            text_blocks.sort(key=lambda b: (b[1], b[0]))
            for x0, y0, x1, y1, text, block_type in text_blocks:
                for line in text.split('\n'):
                    if line.strip():
                        out.write(line.strip() + '\n')
            out.write('\n')


def extract_sections_smart(pdf_path: str, output_txt_path: str, min_ratio: float = 0.5):
    """
    Smart section extraction:
      1. Extract sections based on section_patterns
      2. If extracted length < full text length * min_ratio, fallback to full block extraction
    """
    doc = fitz.open(pdf_path)
    # Original full text
    full_text = "".join(page.get_text() for page in doc)
    lines = full_text.split('\n')

    # Match section titles by keywords
    sections = {}
    current_section = "Front Matter"
    buffer = []

    def match_section(line: str):
        low = line.strip().lower()
        for name, pat in section_patterns.items():
            if re.search(pat, low):
                return name
        return None

    for line in lines:
        sec = match_section(line)
        if sec:
            # Save the old section before switching
            if buffer:
                sections.setdefault(current_section, []).append("\n".join(buffer).strip())
            current_section = sec
            buffer = []
        else:
            buffer.append(line)
    # Last section
    if buffer:
        sections.setdefault(current_section, []).append("\n".join(buffer).strip())

    # Concatenate text for each section
    extracted = ""
    for name, parts in sections.items():
        extracted += f"[{name}]\n" + "\n\n".join(parts) + "\n\n"

    # Determine whether to fallback to full text extraction
    if len(extracted) < len(full_text) * min_ratio:
        extract_full_text_by_blocks(pdf_path, output_txt_path)
    else:
        with open(output_txt_path, 'w', encoding='utf-8') as out:
            out.write(extracted)

# Example call
# extract_sections_smart("paper_pdf/YourPaper.pdf", "paper_txt/YourPaper.txt")