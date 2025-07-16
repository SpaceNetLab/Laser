import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_coarse_faiss_index(
    txt_dir: str,
    index_path: str,
    mapping_path: str,
    model_name: str = 'allenai-specter'
):
    """
    Build the first-level (coarse-grained) Faiss database:

      1. Traverse all .txt files in txt_dir, sorted by alphabet/file name
      2. Use SentenceTransformer(model_name) to encode the full text of each paper
      3. Perform L2 normalization on vectors, use IndexFlatIP (inner product approximates cosine similarity) to build the index
      4. Save the index to index_path and the file name list to mapping_path (JSON)
    """
    # 1. Load the model
    model = SentenceTransformer(model_name)

    # 2. Collect all .txt files
    files = sorted([f for f in os.listdir(txt_dir) if f.lower().endswith('.txt')])
    embeddings = []
    file_names = []

    # 3. Generate embeddings
    for fname in files:
        path = os.path.join(txt_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        emb = model.encode(text, convert_to_numpy=True, show_progress_bar=True)
        embeddings.append(emb)
        file_names.append(fname)

    embeddings = np.vstack(embeddings)  # Shape (N, D)

    # 4. L2 normalization (use inner product for cosine similarity)
    faiss.normalize_L2(embeddings)

    # 5. Create and populate the index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    # 6. Save the index and mapping table
    faiss.write_index(index, index_path)
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({'files': file_names}, f, ensure_ascii=False, indent=2)

    print(f"✅ Coarse-grained index saved: {index_path}")
    print(f"✅ File name mapping saved: {mapping_path}")

if __name__ == '__main__':
    build_coarse_faiss_index(
        txt_dir='paper_txt',
        index_path='coarse_index.faiss',
        mapping_path='coarse_mapping.json',
        model_name='allenai-specter'
    )