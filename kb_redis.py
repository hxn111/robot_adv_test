import hashlib
import json
import os
import re
import struct
import time
from pathlib import Path

# import redis


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in ['1', 'true', 'yes', 'on']


def _safe_decode(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    if value is None:
        return ''
    return str(value)


def _clip_text(text, max_len):
    text = str(text or '').strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + '...'


def build_kb_embedding_client():
    provider = str(os.environ.get('KB_EMBEDDINGS_PROVIDER', 'openai')).strip().lower()
    if provider not in ['openai', 'azure', 'auto']:
        provider = 'openai'

    if provider == 'auto':
        # Keep auto conservative: default to OpenAI unless explicitly configured for Azure.
        if os.environ.get('KB_AZURE_OPENAI_API_KEY'):
            provider = 'azure'
        else:
            provider = 'openai'

    if provider == 'azure':
        from openai import AzureOpenAI
        api_key = os.environ.get('KB_AZURE_OPENAI_API_KEY') or os.environ.get('AZURE_OPENAI_API_KEY')
        endpoint = (
            os.environ.get('KB_AZURE_OPENAI_ENDPOINT')
            or os.environ.get('AZURE_OPENAI_ENDPOINT')
            or 'https://cdisrobotdisplay.openai.azure.com'
        )
        api_version = (
            os.environ.get('KB_AZURE_OPENAI_API_VERSION')
            or os.environ.get('AZURE_OPENAI_API_VERSION')
            or '2025-01-01-preview'
        )
        if not api_key:
            raise RuntimeError(
                "KB_EMBEDDINGS_PROVIDER=azure but no KB_AZURE_OPENAI_API_KEY/AZURE_OPENAI_API_KEY found."
            )
        print(f"[KB] Embeddings provider=azure endpoint={endpoint} api_version={api_version}")
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=api_key,
        )

    from openai import OpenAI
    api_key = os.environ.get('KB_OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if api_key:
        print('[KB] Embeddings provider=openai')
        return OpenAI(api_key=api_key)
    print('[KB] Embeddings provider=openai (using ambient auth config)')
    return OpenAI()


def normalize_kb_entry(raw_entry, source_file=''):
    if not isinstance(raw_entry, dict):
        return None

    question = str(raw_entry.get('question') or raw_entry.get('q') or '').strip()
    answer = str(raw_entry.get('answer') or raw_entry.get('a') or '').strip()
    if not question or not answer:
        return None

    kb_id = str(raw_entry.get('id') or raw_entry.get('kb_id') or '').strip()
    if not kb_id:
        seed = f"{question}\n{answer}"
        kb_id = hashlib.sha1(seed.encode('utf-8')).hexdigest()[:16]
    kb_id = re.sub(r'[^a-zA-Z0-9:_-]+', '_', kb_id)

    category = str(raw_entry.get('category') or 'general').strip().lower()
    category = re.sub(r'[^a-z0-9_-]+', '-', category).strip('-') or 'general'

    keywords = raw_entry.get('keywords', [])
    if isinstance(keywords, str):
        keywords = [x.strip() for x in keywords.split(',') if str(x).strip()]
    elif isinstance(keywords, list):
        keywords = [str(x).strip() for x in keywords if str(x).strip()]
    else:
        keywords = []

    alt_questions = (
        raw_entry.get('alt_questions')
        or raw_entry.get('alternate_questions')
        or raw_entry.get('aliases')
        or []
    )
    if isinstance(alt_questions, str):
        alt_questions = [alt_questions.strip()] if alt_questions.strip() else []
    elif isinstance(alt_questions, list):
        alt_questions = [str(x).strip() for x in alt_questions if str(x).strip()]
    else:
        alt_questions = []

    return {
        'kb_id': kb_id,
        'question': question,
        'answer': answer,
        'category': category,
        'keywords': keywords,
        'alt_questions': alt_questions,
        'source_file': source_file,
    }


def load_kb_entries_from_file(file_path):
    path = Path(file_path)
    if not path.is_file():
        return []

    if path.suffix.lower() == '.jsonl':
        entries = []
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    with path.open('r', encoding='utf-8') as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get('entries'), list):
            return payload['entries']
        return [payload]
    return []


def _normalize_input_paths(input_paths):
    if input_paths is None:
        raw = os.environ.get('KB_AUTO_INGEST_INPUT', 'knowledge_base')
        return [x.strip() for x in re.split(r'[;,]', str(raw)) if x.strip()]

    if isinstance(input_paths, (str, Path)):
        return [x.strip() for x in re.split(r'[;,]', str(input_paths)) if x.strip()]

    normalized = []
    for value in input_paths:
        item = str(value or '').strip()
        if not item:
            continue
        normalized.extend([x.strip() for x in re.split(r'[;,]', item) if x.strip()])
    return normalized


def resolve_kb_input_files(input_paths=None):
    files = []
    for raw_path in _normalize_input_paths(input_paths):
        path = Path(raw_path)
        if path.is_file():
            files.append(path)
            continue
        if path.is_dir():
            files.extend(sorted(path.glob('*.json')))
            files.extend(sorted(path.glob('*.jsonl')))

    deduped = []
    seen = set()
    for file_path in files:
        key = str(file_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(file_path)
    return deduped


def _compute_kb_fingerprint(files, kb):
    hasher = hashlib.sha1()
    seed = (
        f"index={kb.index_name}|prefix={kb.key_prefix}|model={kb.embedding_model}"
        f"|dim={kb.embedding_dim}|metric={kb.distance_metric}"
    )
    hasher.update(seed.encode('utf-8'))
    for file_path in sorted(files, key=lambda p: str(p.resolve())):
        resolved = file_path.resolve()
        hasher.update(str(resolved).encode('utf-8'))
        with resolved.open('rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
    return hasher.hexdigest()


def _has_any_docs(client, key_prefix):
    pattern = f"{key_prefix}*"
    for _ in client.scan_iter(match=pattern, count=1):
        return True
    return False


def bootstrap_kb_in_redis(kb, input_paths=None, replace=False, skip_if_unchanged=True):
    if not isinstance(kb, RedisVectorKnowledgeBase):
        raise TypeError('kb must be a RedisVectorKnowledgeBase instance.')

    result = {
        'status': 'disabled',
        'files': [],
        'file_counts': {},
        'upserted': 0,
        'skipped': 0,
        'deleted': 0,
    }
    if not kb.enabled:
        return result

    files = resolve_kb_input_files(input_paths=input_paths)
    result['files'] = [str(p) for p in files]
    if not files:
        result['status'] = 'no_files'
        return result

    client = kb._get_client()
    state_key = os.environ.get('KB_AUTO_INGEST_STATE_KEY', 'kb:faq:ingest:fingerprint')

    kb.ensure_index(drop_if_exists=replace)
    if replace:
        result['deleted'] = kb.clear_docs()
    else:
        try:
            if skip_if_unchanged:
                fingerprint = _compute_kb_fingerprint(files, kb)
                previous = _safe_decode(client.get(state_key)).strip()
                if previous and previous == fingerprint and _has_any_docs(client, kb.key_prefix):
                    result['status'] = 'skipped_unchanged'
                    return result
        except Exception as e:
            print(f"[KB] auto-ingest fingerprint check failed: {e}")

    normalized_entries = []
    file_counts = {}
    for file_path in files:
        raw_entries = load_kb_entries_from_file(file_path)
        count = 0
        for raw_entry in raw_entries:
            entry = normalize_kb_entry(raw_entry, source_file=file_path.name)
            if not entry:
                continue
            normalized_entries.append(entry)
            count += 1
        file_counts[str(file_path)] = count
    result['file_counts'] = file_counts

    if not normalized_entries:
        result['status'] = 'no_valid_entries'
        return result

    stats = kb.upsert_entries(normalized_entries)
    result['upserted'] = int(stats.get('upserted', 0))
    result['skipped'] = int(stats.get('skipped', 0))
    result['status'] = 'ingested'

    try:
        fingerprint = _compute_kb_fingerprint(files, kb)
        client.set(state_key, fingerprint)
    except Exception as e:
        print(f"[KB] auto-ingest state update failed: {e}")

    return result


class RedisVectorKnowledgeBase:
    def __init__(self, embed_client, force_enable=False):
        self.embed_client = embed_client
        self.enabled = force_enable or _env_bool('KB_REDIS_ENABLED', default=False)
        self.host = os.environ.get('KB_REDIS_HOST', '127.0.0.1')
        self.port = int(os.environ.get('KB_REDIS_PORT', 6379))
        self.db = int(os.environ.get('KB_REDIS_DB', 0))
        self.index_name = os.environ.get('KB_REDIS_INDEX_NAME', 'kb:faq_idx')
        self.key_prefix = os.environ.get('KB_REDIS_KEY_PREFIX', 'kb:faq:')
        self.embedding_model = os.environ.get('KB_EMBEDDING_MODEL', 'text-embedding-3-small')
        self.embedding_dim = int(os.environ.get('KB_EMBEDDING_DIM', 1536))
        self.distance_metric = os.environ.get('KB_REDIS_DISTANCE_METRIC', 'COSINE').upper()
        self.top_k = int(os.environ.get('KB_RETRIEVAL_TOP_K', 3))
        self.socket_timeout = float(os.environ.get('KB_REDIS_SOCKET_TIMEOUT_S', 4))
        self._redis = None

    def _get_client(self):
        if self._redis is None:
            self._redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False,
                socket_timeout=self.socket_timeout,
            )
        return self._redis

    def _vector_to_bytes(self, vector):
        if not isinstance(vector, list) or not vector:
            raise ValueError('Embedding vector is empty.')
        if self.embedding_dim and len(vector) != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch. expected={self.embedding_dim} got={len(vector)}"
            )
        return struct.pack(f"<{len(vector)}f", *vector)

    def _embed_text(self, text):
        if not isinstance(text, str) or not text.strip():
            return None
        ans = self.embed_client.embeddings.create(
            model=self.embedding_model,
            input=text.strip()
        )
        if not getattr(ans, 'data', None):
            return None
        return ans.data[0].embedding

    def _build_embedding_text(self, entry):
        parts = [f"Question: {entry.get('question', '')}"]
        alt_questions = entry.get('alt_questions') or []
        if alt_questions:
            parts.append("Related questions: " + " | ".join(alt_questions))
        keywords = entry.get('keywords') or []
        if keywords:
            parts.append("Keywords: " + ", ".join(keywords))
        parts.append(f"Answer: {entry.get('answer', '')}")
        return "\n".join(parts).strip()

    def ensure_index(self, drop_if_exists=False):
        if not self.enabled:
            return

        client = self._get_client()
        index_exists = True
        try:
            client.execute_command('FT.INFO', self.index_name)
        except Exception:
            index_exists = False

        if drop_if_exists and index_exists:
            client.execute_command('FT.DROPINDEX', self.index_name)
            index_exists = False

        if index_exists:
            return

        client.execute_command(
            'FT.CREATE',
            self.index_name,
            'ON', 'HASH',
            'PREFIX', '1', self.key_prefix,
            'SCHEMA',
            'kb_id', 'TEXT',
            'question', 'TEXT',
            'answer', 'TEXT',
            'category', 'TAG',
            'keywords', 'TEXT',
            'source_file', 'TAG',
            'updated_at', 'NUMERIC',
            'embedding', 'VECTOR', 'HNSW', '6',
            'TYPE', 'FLOAT32',
            'DIM', str(self.embedding_dim),
            'DISTANCE_METRIC', self.distance_metric,
        )

    def clear_docs(self):
        if not self.enabled:
            return 0
        client = self._get_client()
        match_pattern = f"{self.key_prefix}*"
        deleted = 0
        batch = []
        for key in client.scan_iter(match=match_pattern, count=500):
            batch.append(key)
            if len(batch) >= 500:
                deleted += int(client.delete(*batch))
                batch = []
        if batch:
            deleted += int(client.delete(*batch))
        return deleted

    def upsert_entries(self, normalized_entries):
        if not self.enabled:
            return {'upserted': 0, 'skipped': len(normalized_entries)}

        client = self._get_client()
        upserted = 0
        skipped = 0
        now_unix = int(time.time())
        for entry in normalized_entries:
            if not isinstance(entry, dict):
                skipped += 1
                continue
            try:
                embedding_text = self._build_embedding_text(entry)
                embedding = self._embed_text(embedding_text)
                if not embedding:
                    skipped += 1
                    continue
                embedding_bytes = self._vector_to_bytes(embedding)

                key = f"{self.key_prefix}{entry['kb_id']}"
                mapping = {
                    'kb_id': entry['kb_id'],
                    'question': entry['question'],
                    'answer': entry['answer'],
                    'category': entry.get('category', 'general'),
                    'keywords': ', '.join(entry.get('keywords') or []),
                    'source_file': entry.get('source_file', ''),
                    'updated_at': str(now_unix),
                    'embedding': embedding_bytes,
                }
                client.hset(key, mapping=mapping)
                upserted += 1
            except Exception as e:
                print(f"[KB] upsert failed for id={entry.get('kb_id')}: {e}")
                skipped += 1

        return {'upserted': upserted, 'skipped': skipped}

    def search(self, query, top_k=None):
        if not self.enabled:
            return []
        if not isinstance(query, str) or not query.strip():
            return []

        top_k = top_k or self.top_k
        if top_k <= 0:
            return []

        try:
            client = self._get_client()
            embedding = self._embed_text(query)
            if not embedding:
                return []
            vector_bytes = self._vector_to_bytes(embedding)

            res = client.execute_command(
                'FT.SEARCH',
                self.index_name,
                f"*=>[KNN {int(top_k)} @embedding $vec AS score]",
                'PARAMS', '2', 'vec', vector_bytes,
                'SORTBY', 'score', 'ASC',
                'RETURN', '8',
                'kb_id', 'question', 'answer', 'category', 'keywords', 'source_file', 'updated_at', 'score',
                'DIALECT', '2',
            )
            return self._parse_search_response(res)
        except Exception as e:
            print(f"[KB] search failed: {e}")
            return []

    def _parse_search_response(self, response):
        if not isinstance(response, list) or len(response) < 2:
            return []

        hits = []
        for i in range(1, len(response), 2):
            if i + 1 >= len(response):
                break
            fields = response[i + 1]
            if not isinstance(fields, list):
                continue
            mapped = {}
            for j in range(0, len(fields), 2):
                if j + 1 >= len(fields):
                    break
                key = _safe_decode(fields[j])
                val = _safe_decode(fields[j + 1])
                mapped[key] = val
            score_str = mapped.get('score', '')
            try:
                score = float(score_str)
            except (TypeError, ValueError):
                score = None
            hits.append({
                'kb_id': mapped.get('kb_id', ''),
                'question': mapped.get('question', ''),
                'answer': mapped.get('answer', ''),
                'category': mapped.get('category', ''),
                'keywords': mapped.get('keywords', ''),
                'source_file': mapped.get('source_file', ''),
                'score': score,
            })
        return hits

    def format_for_prompt(self, hits):
        if not isinstance(hits, list) or not hits:
            return ''

        lines = [
            "## Retrieved Knowledge Base Snippets",
            "Use these only when relevant to the latest user input.",
            "If snippets are irrelevant, ignore them and respond normally.",
            ""
        ]
        for idx, hit in enumerate(hits, start=1):
            kb_id = str(hit.get('kb_id', '')).strip() or f"hit-{idx}"
            question = _clip_text(hit.get('question', ''), 240)
            answer = _clip_text(hit.get('answer', ''), 700)
            category = str(hit.get('category', '')).strip()
            lines.append(f"{idx}. [{kb_id}] category={category or 'general'}")
            lines.append(f"Q: {question}")
            lines.append(f"A: {answer}")
            lines.append("")
        return '\n'.join(lines).strip()
