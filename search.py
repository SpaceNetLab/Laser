import os
import re
import json
import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

def encode_query(query: str, model_name: str = 'allenai/specter') -> np.ndarray:
    """
    Encode the query text into a vector:
      - Use the output at the [CLS] position
      - Truncate to the model's maximum position_embeddings length
      - Perform L2 normalization
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    max_len = model.config.max_position_embeddings
    enc = tokenizer(
        query,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    with torch.no_grad():
        out = model(**enc, return_dict=True)
        vec = out.last_hidden_state[:, 0]  # (1, D)
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
    return vec.cpu().numpy()  # (1, D)

def load_paragraphs(txt_dir: str) -> list:
    """
    Read all .txt files from txt_dir in order,
    split them into paragraphs by empty lines, and return the list of paragraphs.
    """
    paras = []
    for fname in sorted(os.listdir(txt_dir)):
        if not fname.lower().endswith('.txt'):
            continue
        text = open(os.path.join(txt_dir, fname), 'r', encoding='utf-8').read()
        chunks = re.split(r'\n\s*\n+', text)
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk:
                paras.append(chunk)
    return paras

def retrieve_topk(
    query: str,
    index_path: str = 'fine_index.faiss',
    txt_dir: str = 'paper_txt',
    output_path: str = 'results.txt',
    top_k: int = 25,
    model_name: str = 'allenai-specter'
):
    # 1. Load the FAISS index
    index = faiss.read_index(index_path)

    # 2. Encode the query
    q_vec = encode_query(query, model_name)

    # 3. Search for top_k
    D, I = index.search(q_vec, top_k)
    indices = I[0]

    # 4. Load the list of paragraphs
    paras = load_paragraphs(txt_dir)

    # 5. Write the results
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Original Query:\n" + query + "\n\n")
        f.write(f"Top {top_k} Relevant Paragraphs:\n\n")
        for idx in indices:
            f.write(paras[idx] + "\n\n")

    print(f"✅ Retrieval complete! Results saved to {output_path}")

if __name__ == '__main__':
    # Example usage: Replace the string below with your original query
    user_query = "How does network topology impact synchronization in LEO satellite networks?"
    retrieve_topk(
        query=user_query,
        index_path='fine_index.faiss',
        txt_dir='paper_txt',
        output_path='retrieval_results.txt',
        top_k=25,
        model_name='allenai-specter'
    )