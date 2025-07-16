import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

###############################################################################
# 0) Pre-configuration: Index and mapping file names, model, output
###############################################################################
DOC_INDEX_PATH = "doc_index.index"       # Document-level index (CPU)
DOC_META_PATH  = "doc_name_list.pkl"     # Document name list

PARA_INDEX_PATH = "paragraph_index.index"  # Paragraph-level index (CPU)
PARA_META_PATH  = "paragraph_info.pkl"     # [(pdf_name, paragraph_text), ...]

OUTPUT_TXT = "final_25_paras.txt"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Hardcoded query question in the code
USER_QUESTION = "I want study how Satellite ISL method affect delay."

###############################################################################
# 1) Load indexes and mappings on CPU
###############################################################################
print("[Info] Loading document-level index (CPU)...")
doc_index = faiss.read_index(DOC_INDEX_PATH)
with open(DOC_META_PATH, "rb") as f:
    doc_name_list = pickle.load(f)
print(f"[Info] Document-level index loaded: {doc_index.ntotal} documents indexed.")

print("[Info] Loading paragraph-level index (CPU)...")
para_index = faiss.read_index(PARA_INDEX_PATH)
with open(PARA_META_PATH, "rb") as f:
    paragraph_info_list = pickle.load(f)
print(f"[Info] Paragraph-level index loaded: {para_index.ntotal} paragraphs indexed.")

###############################################################################
# 2) Load Sentence-BERT model
###############################################################################
# If you have GPU-enabled PyTorch installed and want to accelerate encoding, use 'cuda'; otherwise, use 'cpu'
device_for_encode = "cuda"  # or "cpu"
model = SentenceTransformer(MODEL_NAME, device=device_for_encode)
print(f"[Info] Model loaded: {MODEL_NAME}, device={device_for_encode}")

###############################################################################
# 3) Define retrieval functions
###############################################################################
def search_top_docs(question, index, doc_names, top_k=5):
    """
    Retrieve the top_k most relevant documents from the document-level index (CPU).
    Returns: [(doc_name, distance, doc_id), ...]
    """
    query_vec = model.encode(question, device=device_for_encode)
    query_vec = np.array([query_vec]).astype('float32')  # shape: (1, dim)
    distances, ids = index.search(query_vec, top_k)      # Search on CPU
    results = []
    for dist, idx in zip(distances[0], ids[0]):
        results.append((doc_names[idx], dist, idx))
    return results

def search_top_paragraphs_in_one_doc(question, doc_name, para_index, paragraph_info, top_k=5):
    """
    In the paragraph-level "main index", find paragraphs belonging to doc_name -> build a CPU sub-index -> retrieve top_k paragraphs on CPU.
    Returns: [(paragraph_text, distance), ...]
    """
    # 1) Find all paragraphs corresponding to doc_name
    doc_chunk_ids = [i for i, (fn, _) in enumerate(paragraph_info) if fn == doc_name]
    if not doc_chunk_ids:
        return []

    # 2) Reconstruct vectors for these paragraphs and add them to a sub-index
    dim = para_index.d
    sub_index = faiss.IndexFlatIP(dim)  # CPU index

    sub_embeddings = []
    for gid in doc_chunk_ids:
        emb = para_index.reconstruct(gid)  # Reconstruct on CPU index
        sub_embeddings.append(emb)
    sub_embeddings = np.array(sub_embeddings).astype('float32')
    sub_index.add(sub_embeddings)

    local_id_to_global_id = doc_chunk_ids

    # 3) Encode query vector
    query_vec = model.encode(question, device=device_for_encode)
    query_vec = np.array([query_vec]).astype('float32')

    # 4) Search top_k in the sub-index
    distances, ids = sub_index.search(query_vec, top_k)

    # 5) Parse results
    results = []
    for dist, local_id in zip(distances[0], ids[0]):
        global_id = local_id_to_global_id[local_id]
        para_text = paragraph_info[global_id][1]  # (pdf_name, paragraph_text)
        results.append((para_text, dist))
    return results

###############################################################################
# 4) Two-level retrieval
###############################################################################
print(f"\n[Query] {USER_QUESTION}")
# First level: Retrieve top-5 documents from the document-level index
top_docs = search_top_docs(USER_QUESTION, doc_index, doc_name_list, top_k=5)
print("[Result] Document-level retrieval Top-5:")
for doc_name, score, doc_id in top_docs:
    print(f"  - {doc_name}, similarity={score:.4f}")

# Second level: Retrieve top-5 paragraphs for each document (5x5 = 25)
all_paragraphs = []
for doc_name, doc_score, doc_id in top_docs:
    top_paras = search_top_paragraphs_in_one_doc(
        USER_QUESTION, doc_name, para_index, paragraph_info_list, top_k=5
    )
    all_paragraphs.append((doc_name, doc_score, top_paras))

###############################################################################
# 5) Write the final results (original question + 25 paragraphs) to a txt file
###############################################################################
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    # Write the original question
    f.write("[User Question]\n")
    f.write(USER_QUESTION + "\n\n")

    # Write the retrieved documents and paragraphs
    f.write("[Two-level Retrieval: Document-level Top-5 + Paragraph-level Top-5 per document]\n")
    for doc_name, doc_score, para_list in all_paragraphs:
        f.write(f"=== Document: {doc_name} (similarity={doc_score:.4f}) ===\n")
        for i, (para_text, dist) in enumerate(para_list, start=1):
            f.write(f"{i}) Paragraph similarity={dist:.4f}\n")
            f.write(f"{para_text}\n\n")

    f.write("[END]\n")

print(f"\n[Info] 25 paragraphs (5x5) written to: {OUTPUT_TXT}")