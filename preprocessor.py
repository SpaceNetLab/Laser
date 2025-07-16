import fitz  # The import name for PyMuPDF is fitz
import os


def pdf_to_text_pymupdf(pdf_path):
    """
    Parse the PDF using PyMuPDF and return the extracted text.
    If the PDF is a scanned document without a text layer, this may return an empty string.
    """
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            # get_text("text") parameter defaults to reading the text layer
            page_text = page.get_text()
            text += page_text + "\n"
    return text


def batch_pdf_to_txt_pymupdf(pdf_folder, txt_folder):
    """
    Batch parse all PDF files in the pdf_folder into text and store them in txt_folder.
    The output text files will have the same name as the PDF files, with the extension changed to .txt.
    """
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)

            try:
                extracted_text = pdf_to_text_pymupdf(pdf_path)
            except Exception as e:
                print(f"Failed to parse: {pdf_path}, Error: {e}")
                continue

            base_name = os.path.splitext(filename)[0]
            txt_filename = base_name + '.txt'
            txt_path = os.path.join(txt_folder, txt_filename)

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)

            print(f"Conversion completed: {pdf_path} -> {txt_path}")


if __name__ == "__main__":
    input_pdf_dir = "./paper_pdf"  # Your PDF folder
    output_txt_dir = "./paper_txt"  # Directory to store the parsed txt files

    batch_pdf_to_txt_pymupdf(input_pdf_dir, output_txt_dir)