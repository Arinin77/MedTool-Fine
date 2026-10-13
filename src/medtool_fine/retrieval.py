"""Final MedTool-Fine research implementation."""
from collections import Counter
from pathlib import Path
import importlib.metadata
import math
import re
from .io import read_json, write_json, sha, seal, verify

ROOT = Path(__file__).resolve().parents[2]

def query_text(case, observations=()):
    """Only explicit public fields, never arbitrary JSON serialization."""
    goal, body = (case['clinical_goal'], case['case_text'])
    if not isinstance(goal, str) or not isinstance(body, str) or (not goal.strip()) or (not body.strip()):
        raise ValueError('Nonempty clinical_goal and case_text required')
    if isinstance(observations, str) or any((not isinstance(x, str) for x in observations)):
        raise ValueError('observations must be received answer strings')
    return goal + '\n' + body + ('\nReceived information:\n' + '\n'.join(observations) if observations else '')

def token_windows(token_ids, capacity=510, overlap=64):
    if not 0 <= overlap < capacity:
        raise ValueError('Invalid token window configuration')
    if not token_ids:
        raise ValueError('Empty token sequence')
    result, start, covered = ([], 0, 0)
    while start < len(token_ids):
        end = min(start + capacity, len(token_ids))
        result.append({'start': start, 'end': end, 'weight': end - covered, 'ids': token_ids[start:end]})
        covered = end
        if end == len(token_ids):
            break
        start = end - overlap
    return result

def tokenizer_for(config, side):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config[side + '_model'], revision=config[side + '_revision'], cache_dir=config['cache_dir'], local_files_only=True, use_fast=True)

def prepare_windows(texts, tokenizer, config):
    special = tokenizer.num_special_tokens_to_add(pair=False)
    capacity = config['max_length'] - special
    chunks, summary = ([], [])
    for item_id, text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        windows = token_windows(ids, capacity, config['overlap_tokens'])
        summary.append({'id': item_id, 'token_count_with_special': len(ids) + special, 'chunks': len(windows), 'contains_cjk': bool(re.search('[\\u4e00-\\u9fff]', text)), 'unknown_tokens': ids.count(tokenizer.unk_token_id)})
        for w in windows:
            chunks.append(dict(w, item_id=item_id, input_ids=tokenizer.build_inputs_with_special_tokens(w['ids'])))
    return (chunks, summary)

def documents_from_snapshot(snapshot):
    manifest_path = snapshot / 'index_manifest.json'
    rows = read_json(manifest_path)['entries']
    docs = []
    seen = set()
    for row in rows:
        if row['tool_id'] in seen:
            raise ValueError('Duplicate tool_id')
        path = (snapshot / row['path']).resolve()
        if not path.is_relative_to((snapshot / 'index_text').resolve()):
            raise ValueError('Only frozen index_text is accepted')
        if sha(path) != row['sha256']:
            raise ValueError('Frozen text checksum mismatch')
        docs.append({'tool_id': row['tool_id'], 'text': path.read_text(encoding='utf-8'), 'source_path': row['path'], 'sha256': row['sha256']})
        seen.add(row['tool_id'])
    return docs

def encode_chunks(chunks, tokenizer, config, side):
    import numpy as np
    import torch
    from transformers import AutoModel
    torch.set_num_threads(config['cpu_threads'])
    torch.manual_seed(0)
    model = AutoModel.from_pretrained(config[side + '_model'], revision=config[side + '_revision'], cache_dir=config['cache_dir'], local_files_only=True, use_safetensors=True).eval()
    if config['max_length'] > model.config.max_position_embeddings:
        raise ValueError('Token window exceeds model capacity')
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(chunks), config['batch_size']):
            batch = chunks[start:start + config['batch_size']]
            inputs = tokenizer.pad([{'input_ids': w['input_ids'], 'attention_mask': [1] * len(w['input_ids'])} for w in batch], padding=True, return_tensors='pt')
            outputs.append(model(**inputs).last_hidden_state[:, 0, :].cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite embeddings')
    return result

def rank_scores(scores):
    return [dict(tool_id=tid, score=float(value), rank=i) for i, (tid, value) in enumerate(sorted(scores.items(), key=lambda x: (-x[1], x[0])), 1)]

def lexical_rank(text, documents, k1=1.2, b=0.75):
    tokenize = lambda s: re.findall('[a-z0-9]+', s.lower())
    terms = {key: Counter(tokenize(value)) for key, value in documents.items()}
    lengths = {key: sum(value.values()) for key, value in terms.items()}
    avg = sum(lengths.values()) / len(lengths) or 1
    df = Counter((word for doc in terms.values() for word in doc))
    query = sorted(set(tokenize(text)))
    scores = {}
    for key, doc in terms.items():
        scores[key] = sum((math.log(1 + (len(terms) - df[word] + 0.5) / (df[word] + 0.5)) * doc[word] * (k1 + 1) / (doc[word] + k1 * (1 - b + b * lengths[key] / avg)) for word in query if doc[word]))
    return rank_scores(scores)

def dense_rank(query_vectors, query_chunks, doc_vectors, doc_chunks, tool_ids):
    import numpy as np
    pair_scores = query_vectors @ doc_vectors.T
    weights = np.array([w['weight'] for w in query_chunks], dtype=float)
    scores = {}
    for tid in tool_ids:
        columns = [i for i, row in enumerate(doc_chunks) if row['item_id'] == tid]
        scores[tid] = float(np.average(pair_scores[:, columns].max(axis=1), weights=weights))
    return rank_scores(scores)

def rrf_rank(rankings, constant=60):
    if constant <= 0:
        raise ValueError('Positive RRF constant required')
    scores = {}
    for ranking in rankings:
        if len({row['tool_id'] for row in ranking}) != len(ranking):
            raise ValueError('Duplicate IDs in ranking')
        for i, row in enumerate(ranking, 1):
            scores[row['tool_id']] = scores.get(row['tool_id'], 0) + 1 / (constant + i)
    return rank_scores(scores)

def build_index(config_path, output):
    import numpy as np
    config = read_json(config_path)
    if output.exists():
        raise FileExistsError(output)
    snapshot = ROOT / config['snapshot']
    docs = documents_from_snapshot(snapshot)
    tok = tokenizer_for(config, 'document')
    chunks, lengths = prepare_windows([(d['tool_id'], d['text']) for d in docs], tok, config)
    vectors = encode_chunks(chunks, tok, config, 'document')
    output.mkdir(parents=True)
    write_json(output / 'config.json', config)
    write_json(output / 'documents.json', docs)
    write_json(output / 'document_chunks.json', [{k: v for k, v in row.items() if k not in ('ids', 'input_ids')} for row in chunks])
    write_json(output / 'token_lengths.json', lengths)
    np.save(output / 'document_vectors.npy', vectors, allow_pickle=False)
    write_json(output / 'manifest.json', {'protocol': config['protocol'], 'tool_count': len(docs), 'chunk_count': len(chunks), 'dimensions': int(vectors.shape[1]), 'source_index_manifest_sha256': sha(snapshot / 'index_manifest.json'), 'code_sha256': sha(Path(__file__)), 'device': 'cpu', 'dtype': 'float32', 'pooling': 'CLS', 'similarity': 'inner_product', 'packages': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers', 'numpy')}, 'generative_api_calls': 0, 'baseline_runs': 0, 'top_k_optimized': False})
    seal(output)
    return read_json(output / 'manifest.json')

class Retriever:

    def __init__(self, directory):
        import numpy as np
        self.directory = Path(directory)
        verify(self.directory)
        self.config = read_json(self.directory / 'config.json')
        self.docs = read_json(self.directory / 'documents.json')
        self.chunks = read_json(self.directory / 'document_chunks.json')
        self.vectors = np.load(self.directory / 'document_vectors.npy', allow_pickle=False)
        self.tokenizer = tokenizer_for(self.config, 'query')

    def retrieve(self, case, observations=(), top_k=None):
        k = self.config['default_top_k'] if top_k is None else top_k
        if not isinstance(k, int) or isinstance(k, bool) or (not 1 <= k <= len(self.docs)):
            raise ValueError('top_k out of range')
        text = query_text(case, observations)
        chunks, lengths = prepare_windows([('query', text)], self.tokenizer, self.config)
        embeddings = encode_chunks(chunks, self.tokenizer, self.config, 'query')
        dense = dense_rank(embeddings, chunks, self.vectors, self.chunks, [d['tool_id'] for d in self.docs])
        bm25 = lexical_rank(text, {d['tool_id']: d['text'] for d in self.docs}, self.config['bm25_k1'], self.config['bm25_b'])
        hybrid = rrf_rank([bm25, dense], self.config['rrf_constant'])
        return {'query_text': text, 'token_length': lengths[0], 'top_k': k, 'bm25': bm25[:k], 'dense': dense[:k], 'hybrid': hybrid[:k]}

