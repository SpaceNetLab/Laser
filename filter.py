import re

def clean_text(text: str) -> str:
    """
    Clean and format text content:
    1. Remove numeric references (e.g., [1], [0, 2, 4], etc.)
    2. Remove all other square brackets and their contents (e.g., [HS08], [OFS08], etc.)
    3. Merge paragraph breaks caused by hyphenated line breaks
    4. Replace all line breaks within the same paragraph with spaces, outputting each paragraph as a single line
    """
    # Remove numeric references
    text = re.sub(r'\[[\d,\s]+\]', '', text)
    # Remove other square brackets and their contents
    text = re.sub(r'\[[^\]]+\]', '', text)
    # Merge hyphenated line breaks
    text = re.sub(r'-\r?\n\s*', '', text)

    # Split paragraphs by empty lines
    paras = re.split(r'\n\s*\n+', text)
    cleaned_paras = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        # Replace all line breaks within a paragraph with spaces
        single_line = re.sub(r'\s*\n\s*', ' ', p)
        cleaned_paras.append(single_line)
    # Separate paragraphs with two line breaks (can be adjusted to one if needed)
    return "\n\n".join(cleaned_paras)

def clean_file(input_path: str, output_path: str):
    """
    Read input_path, clean and format the content, then write to output_path
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    cleaned = clean_text(content)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

if __name__ == "__main__":
    # Example call: Clean and format 'output.txt' into 'cleaned_output.txt'
    clean_file('output.txt', 'cleaned_output.txt')