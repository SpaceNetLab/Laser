import os
import re
import json
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file

def encode_texts(texts, model_name='allenai-specter', batch_size=8):
    """
    Use HuggingFace Transformers to vectorize a series of texts:
      - Extract the [CLS] token vector from the model output
      - Perform L2 normalization
    """
    model_name = "allenai/specter"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    max_len = model.config.max_position_embeddings

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors='pt'
            )
            out = model(**enc, return_dict=True)
            cls_emb = out.last_hidden_state[:, 0]  # (B, D)
            cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=1)  # L2 normalization
            all_embs.append(cls_emb.cpu().numpy())

    return np.vstack(all_embs)

def build_fine_faiss_index(
    txt_dir: str,
    index_path: str,
    mapping_path: str,
    model_name: str = 'allenai-specter',
    batch_size: int = 8
):
    """
    Build a second-level (paragraph-level) Faiss index:
      1. Traverse all .txt files in txt_dir
      2. Split into paragraphs by empty lines and filter out empty paragraphs
      3. Generate embeddings for each paragraph, creating N×D vectors
      4. Build an IndexFlatIP (inner product approximates cosine similarity) index
      5. Save the index file and paragraph mapping table (JSON)
    """
    file_para_map = []
    paras = []

    # Collect all paragraphs
    for fname in sorted(os.listdir(txt_dir)):
        if not fname.lower().endswith('.txt'):
            continue
        path = os.path.join(txt_dir, fname)
        text = open(path, 'r', encoding='utf-8').read()
        # Split by at least one empty line
        chunks = re.split(r'\n\s*\n+', text)
        for pid, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if chunk:
                paras.append(chunk)
                file_para_map.append({
                    'file': fname,
                    'para_id': pid
                })

    # Vectorize
    embeddings = encode_texts(paras, model_name=model_name, batch_size=batch_size)

    # Build Faiss index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    # Write output
    faiss.write_index(index, index_path)
    with open(mapping_path, 'w', encoding='utf-8') as fp:
        json.dump(file_para_map, fp, ensure_ascii=False, indent=2)

    print(f"✅ Saved fine-grained index to {index_path}")
    print(f"✅ Saved paragraph mapping to {mapping_path}")

if __name__ == '__main__':
    build_fine_faiss_index(
        txt_dir='paper_txt',
        index_path='fine_index.faiss',
        mapping_path='fine_mapping.json',
        model_name='allenai-specter',
        batch_size=16
    )