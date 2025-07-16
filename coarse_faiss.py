import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def build_and_save_index(txt_dir, index_path, meta_path,
                         model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Read all .txt files in the txt_dir directory, use Sentence-BERT to generate embeddings on GPU,
    then build a Faiss index on CPU and save the index and document name mapping to disk.

    Parameters:
    - txt_dir   : Folder containing .txt files, each file corresponds to a PDF conversion result
    - index_path: File name to save the index, e.g., 'doc_index.index'
    - meta_path : File name to save the document name list, e.g., 'doc_name_list.pkl'
    - model_name: Name of the sentence-transformers model
    """

    # 1) Load the Sentence-BERT model, specify inference on GPU (device='cuda')
    #    If you don't want to use GPU, change 'cuda' to 'cpu'
    model = SentenceTransformer(model_name, device='cuda')
    print(f"[Info] Model '{model_name}' loaded, encoding on GPU...")

    doc_name_list = []
    embeddings = []

    # 2) Iterate through all .txt files in txt_dir and vectorize the content
    for filename in os.listdir(txt_dir):
        if filename.lower().endswith(".txt"):
            txt_path = os.path.join(txt_dir, filename)
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            if not text:
                print(f"[Warning] File {filename} is empty, skipped.")
                continue

            # Compute embeddings using Sentence-BERT on GPU
            emb = model.encode(text)  # shape: (embedding_dim,)
            embeddings.append(emb)
            doc_name_list.append(filename)

    # 3) Build a Faiss index on CPU (using inner product IP as an example)
    embeddings = np.array(embeddings).astype('float32')  # shape: (num_docs, dim)
    dim = embeddings.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(embeddings)
    print(f"[Info] CPU index built, {cpu_index.ntotal} documents indexed.")

    # 4) Save the index and document name list to disk
    #    Note: Faiss write_index only supports CPU indices, GPU indices need to be converted back to CPU before saving
    faiss.write_index(cpu_index, index_path)
    print(f"[Info] Faiss index saved to: {index_path}")

    with open(meta_path, 'wb') as f:
        pickle.dump(doc_name_list, f)
    print(f"[Info] Document name mapping saved to: {meta_path}")


if __name__ == "__main__":
    # Path to the folder containing PDF->TXT files
    txt_directory = "./paper_txt"

    # Prepare file names for saving the index and mapping
    index_file = "doc_index.index"  # It's better to add a .index suffix
    meta_file = "doc_name_list.pkl"  # Pickle file name, customizable

    # Call the function to build and save the index
    build_and_save_index(txt_directory, index_file, meta_file,
                         model_name='sentence-transformers/all-MiniLM-L6-v2')