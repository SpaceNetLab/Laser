import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def split_text_into_paragraphs(text, min_length=10):
    """
    A simple example of splitting text into paragraphs by line breaks or empty lines.
    You can modify it to use more complex chunking strategies, such as fixed-length chunks or sentence-based splitting.

    Parameters:
    - text: The string of the entire document
    - min_length: Minimum character count to control, paragraphs that are too short may be skipped
    Returns:
    - A list of paragraphs [paragraph1, paragraph2, ...]
    """
    # Split into paragraphs by line breaks (or empty lines)
    raw_paras = text.split('\n')

    paragraphs = []
    for para in raw_paras:
        para = para.strip()
        if len(para) >= min_length:
            paragraphs.append(para)
    return paragraphs


def build_and_save_paragraph_index(txt_dir, index_path, meta_path,
                                   model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    1. Split each .txt file in txt_dir into "paragraphs";
    2. Compute vectors for each paragraph using Sentence-BERT on GPU (or CPU);
    3. Store all paragraph vectors in a FAISS CPU index, then save the index + paragraph information to disk.
    """
    # 1) Load the Sentence-BERT model (using GPU acceleration)
    model = SentenceTransformer(model_name, device='cuda')
    print(f"[Info] Model '{model_name}' loaded, encoding on GPU...")

    # Used to store metadata for all paragraphs: [(pdf_name, paragraph_text), ...]
    chunk_info_list = []
    # Used to store vectors for all paragraphs
    embeddings = []

    # 2) Iterate through the folder, load, split, and vectorize each file
    for filename in os.listdir(txt_dir):
        if filename.lower().endswith(".txt"):
            txt_path = os.path.join(txt_dir, filename)
            with open(txt_path, 'r', encoding='utf-8') as f:
                full_text = f.read().strip()

            if not full_text:
                print(f"[Warning] File {filename} is empty, skipped")
                continue

            # (Example) Split into paragraphs
            paragraphs = split_text_into_paragraphs(full_text, min_length=10)
            if not paragraphs:
                print(f"[Warning] File {filename} has no valid paragraphs, skipped")
                continue

            # Encode each paragraph
            for para_text in paragraphs:
                emb = model.encode(para_text)  # shape: (embedding_dim,)
                embeddings.append(emb)
                chunk_info_list.append((filename, para_text))

    # 3) Build Faiss CPU index
    embeddings = np.array(embeddings).astype('float32')  # shape: (num_paragraphs, dim)
    dim = embeddings.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(embeddings)

    print(f"[Info] Paragraph-level index built, total paragraphs indexed: {cpu_index.ntotal}")

    # 4) Save the index (CPU) and paragraph mapping table
    faiss.write_index(cpu_index, index_path)
    print(f"[Info] Paragraph-level index saved to: {index_path}")

    with open(meta_path, 'wb') as f:
        pickle.dump(chunk_info_list, f)
    print(f"[Info] Paragraph information mapping saved to: {meta_path}")


if __name__ == "__main__":
    # Directory containing your txt files
    txt_directory = "./paper_txt"

    # Index file and paragraph mapping file to save
    paragraph_index_file = "paragraph_index.index"
    paragraph_meta_file = "paragraph_info.pkl"

    build_and_save_paragraph_index(txt_directory,
                                   paragraph_index_file,
                                   paragraph_meta_file,
                                   model_name='sentence-transformers/all-MiniLM-L6-v2')