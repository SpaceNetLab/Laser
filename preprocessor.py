import os
from filter import clean_file
from extractor import *

def preprocess(input_pdf: str, output_txt: str):
    """
    Perform the following steps for a single paper:
      1. Extract text from input_pdf to output_txt
      2. Clean output_txt (overwrite the file)
    """
    extract_sections_smart(input_pdf, output_txt)
    clean_file(output_txt, output_txt)

def main():
    pdf_root = 'paper_pdf'   # Root directory containing all PDFs
    txt_root = 'paper_txt'   # Target directory for cleaned TXT output

    os.makedirs(txt_root, exist_ok=True)

    for dirpath, _, filenames in os.walk(pdf_root):
        for fname in filenames:
            if not fname.lower().endswith('.pdf'):
                continue

            input_pdf = os.path.join(dirpath, fname)
            base = os.path.splitext(fname)[0]
            output_txt = os.path.join(txt_root, f'{base}.txt')

            try:
                preprocess(input_pdf, output_txt)
            except Exception:
                # Skip the file if an error occurs
                pass

if __name__ == '__main__':
    main()