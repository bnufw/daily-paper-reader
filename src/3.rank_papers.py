#!/usr/bin/env python
# 使用 Gemini 评分或兼容旧网关的 rerank 接口对候选论文做重排序。

import argparse
import json
import os
import random
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from llm import GeminiClient, resolve_gemini_api_key, resolve_model_env
except ImportError:
    from llm import BltClient as GeminiClient  # type: ignore

    def resolve_gemini_api_key() -> str:
        return (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("BLT_API_KEY")
            or os.getenv("LLM_API_KEY")
            or ""
        )

    def resolve_model_env(primary_name: str, legacy_name: str, default: str) -> str:
        return os.getenv(primary_name) or os.getenv(legacy_name) or default


SCRIPT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
TODAY_STR = str(os.getenv("DPR_RUN_DATE") or "").strip() or datetime.now(timezone.utc).strftime("%Y%m%d")
ARCHIVE_DIR = os.path.join(ROOT_DIR, "archive", TODAY_STR)
FILTERED_DIR = os.path.join(ARCHIVE_DIR, "filtered")
RANKED_DIR = os.path.join(ARCHIVE_DIR, "rank")

MAX_CHARS_PER_DOC = 850
BATCH_SIZE = 100
TOKEN_SAFETY = 29000
RRF_K = 60
LANE_TOP_K_BASE = 30
LANE_TOP_K_STEP = 10
LANE_TOP_K_MAX = 120
GLOBAL_POOL_GUARANTEED_MIN = 5
GLOBAL_POOL_GUARANTEED_MAX = 20
GLOBAL_POOL_RRF_MIN = 60
GLOBAL_POOL_RRF_MAX = 300


def strip_json_wrappers(text: str) -> str:
  cleaned = str(text or '').strip()
  cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
  cleaned = re.sub(r'\s*```$', '', cleaned)
  return cleaned.strip()


def repair_json_suffix(text: str) -> str:
  if not text:
    return text
  stack: List[str] = []
  in_str = False
  escaped = False
  for ch in text:
    if in_str:
      if escaped:
        escaped = False
        continue
      if ch == '\\':
        escaped = True
        continue
      if ch == '"':
        in_str = False
      continue
    if ch == '"':
      in_str = True
    elif ch == '{':
      stack.append('}')
    elif ch == '[':
      stack.append(']')
    elif ch in ('}', ']'):
      if stack and stack[-1] == ch:
        stack.pop()
  repaired = text
  if in_str:
    repaired += '"'
  if stack:
    repaired += ''.join(reversed(stack))
  repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
  return repaired


def parse_llm_json(content: str) -> Dict[str, Any] | List[Any] | None:
  raw = strip_json_wrappers(content)
  if not raw:
    return None
  candidates: List[str] = []
  decoder = json.JSONDecoder()
  start = raw.find('{')
  end = raw.rfind('}')
  if start != -1:
    candidates.append(raw[start:])
    if end != -1 and end > start:
      candidates.append(raw[start:end + 1])
  else:
    candidates.append(raw)
  seen = set()
  last_exc: Exception | None = None
  for candidate in candidates:
    if candidate in seen:
      continue
    seen.add(candidate)
    try:
      obj, _idx = decoder.raw_decode(candidate)
      if isinstance(obj, (dict, list)):
        return obj
    except Exception as exc:
      last_exc = exc
      repaired = repair_json_suffix(candidate)
      if repaired != candidate:
        try:
          obj = json.loads(repaired)
          if isinstance(obj, (dict, list)):
            return obj
        except Exception as exc2:
          last_exc = exc2
  if last_exc:
    raise last_exc
  return None


def build_score_prompt(query: str, documents: List[str]) -> List[Dict[str, str]]:
  doc_lines = []
  for idx, doc in enumerate(documents):
    doc_lines.append(f"[{idx}]\n{doc}")
  docs_block = "\n\n".join(doc_lines)
  return [
    {
      'role': 'system',
      'content': (
        'You are a paper ranking assistant. '
        'Score each candidate paper against the query on a 0-100 scale. '
        'Higher score means stronger match to the query intent. '
        'Return strict JSON only.'
      ),
    },
    {
      'role': 'user',
      'content': (
        f'Query:\n{query}\n\n'
        'Candidates:\n'
        f'{docs_block}\n\n'
        'Return JSON in this format:\n'
        '{"results":[{"index":0,"score":83}]}\n'
        'Rules:\n'
        '- Every candidate index must appear exactly once.\n'
        '- score must be a number between 0 and 100.\n'
        '- Do not add markdown fences or extra text.'
      ),
    },
  ]


def parse_score_results(content: str, expected_count: int) -> List[Dict[str, float]]:
  payload = parse_llm_json(content)
  if not isinstance(payload, dict):
    raise ValueError('score payload must be a JSON object')
  results = payload.get('results')
  if not isinstance(results, list):
    raise ValueError('score payload missing results list')
  parsed: List[Dict[str, float]] = []
  seen: set[int] = set()
  for item in results:
    if not isinstance(item, dict):
      continue
    try:
      idx = int(item.get('index'))
      score = float(item.get('score'))
    except Exception as exc:
      raise ValueError(f'invalid score item: {item}') from exc
    if idx < 0 or idx >= expected_count:
      raise ValueError(f'index out of range: {idx}')
    if idx in seen:
      raise ValueError(f'duplicate index: {idx}')
    seen.add(idx)
    parsed.append({'index': idx, 'score': max(0.0, min(score, 100.0))})
  if len(parsed) != expected_count:
    raise ValueError(f'incomplete score results: expected={expected_count} actual={len(parsed)}')
  parsed.sort(key=lambda item: item['index'])
  return parsed


def fallback_batch_scores(batch_docs: List[str]) -> List[Dict[str, float]]:
  total = max(len(batch_docs), 1)
  return [
    {'index': idx, 'score': float(total - idx)}
    for idx, _doc in enumerate(batch_docs)
  ]


def score_documents_with_llm(
  client: GeminiClient,
  query: str,
  documents: List[str],
) -> List[Dict[str, float]]:
  messages = build_score_prompt(query, documents)
  last_error: Exception | None = None
  for response_format in ({'type': 'json_object'}, None):
    try:
      response = client.chat(messages=messages, response_format=response_format)
      return parse_score_results(response.get('content') or '', len(documents))
    except Exception as exc:
      last_error = exc
  if last_error is not None:
    raise last_error
  raise RuntimeError('score_documents_with_llm failed without response')


def log(message: str) -> None:
  ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
  print(f"[{ts}] {message}", flush=True)


def group_start(title: str) -> None:
  print(f"::group::{title}", flush=True)


def group_end() -> None:
  print("::endgroup::", flush=True)

def build_token_encoder():
  try:
    import tiktoken  # type: ignore
    return tiktoken.get_encoding("cl100k_base")
  except Exception:
    return None


def estimate_tokens(text: str, encoder) -> int:
  if encoder is None:
    return max(1, len(text) // 3)
  return len(encoder.encode(text))


def score_to_stars(score: float) -> int:
  if score >= 0.9:
    return 5
  if score >= 0.5:
    return 4
  if score >= 0.1:
    return 3
  if score >= 0.01:
    return 2
  return 1


def load_json(path: str) -> Dict[str, Any]:
  if not os.path.exists(path):
    raise FileNotFoundError(f"找不到文件：{path}")
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
  log(f"[INFO] 已将打分结果写入：{path}")


def format_doc(title: str, abstract: str) -> str:
  content = f"Title: {title}\nAbstract: {abstract}".strip()
  if len(content) > MAX_CHARS_PER_DOC:
    content = content[:MAX_CHARS_PER_DOC]
  return content


def build_documents(papers_by_id: Dict[str, Dict[str, Any]], paper_ids: List[str]) -> List[str]:
  docs: List[str] = []
  for pid in paper_ids:
    p = papers_by_id.get(pid)
    if not p:
      docs.append(f"[Missing paper {pid}]")
      continue
    title = (p.get("title") or "").strip()
    abstract = (p.get("abstract") or "").strip()
    if title or abstract:
      docs.append(format_doc(title, abstract))
    else:
      docs.append(f"[Empty paper {pid}]")
  return docs


def get_top_ids(query_obj: Dict[str, Any]) -> List[str]:
  sim_scores = query_obj.get("sim_scores") or {}
  top_ids = query_obj.get("top_ids") or []
  if not top_ids and isinstance(sim_scores, dict) and sim_scores:
    top_ids = sorted(sim_scores.keys(), key=lambda pid: sim_scores[pid].get("rank", 1e9))
  return list(top_ids)


def _unique_keep_order(items: List[str]) -> List[str]:
  seen = set()
  out: List[str] = []
  for item in items:
    pid = str(item or "").strip()
    if not pid or pid in seen:
      continue
    seen.add(pid)
    out.append(pid)
  return out


def _clamp_int(value: float | int, min_value: int, max_value: int) -> int:
  return max(min_value, min(int(value), max_value))


def resolve_global_pool_budget(
  total_papers: int,
  intent_query_count: int,
) -> Tuple[int, int, int]:
  """
  统一候选池预算：
  - lane_top_k 随论文总数递增：1000 篇内 30，每增加 1000 篇 +10，上限 120；
  - guaranteed_per_lane = lane_top_k 的 25%，限制在 [5, 20]；
  - global_rrf_top = lane_top_k * intent_query_count，限制在 [60, 300]。
  """
  total = max(int(total_papers or 0), 0)
  intent_count = max(int(intent_query_count or 0), 1)
  if total <= 0:
    lane_top_k = LANE_TOP_K_BASE
  else:
    blocks = (total - 1) // 1000
    lane_top_k = min(LANE_TOP_K_BASE + LANE_TOP_K_STEP * blocks, LANE_TOP_K_MAX)
  guaranteed_per_lane = _clamp_int(
    round(lane_top_k * 0.25),
    GLOBAL_POOL_GUARANTEED_MIN,
    GLOBAL_POOL_GUARANTEED_MAX,
  )
  global_rrf_top = _clamp_int(
    lane_top_k * intent_count,
    GLOBAL_POOL_RRF_MIN,
    GLOBAL_POOL_RRF_MAX,
  )
  return lane_top_k, guaranteed_per_lane, global_rrf_top


def build_global_candidate_ids(
  queries: List[Dict[str, Any]],
  *,
  guaranteed_per_lane: int,
  global_limit: int,
) -> List[str]:
  """
  将所有 query lane 的候选论文合并成统一候选池。
  - 不区分 keyword / intent_query 来源；
  - 使用 rank-based RRF 做全局聚合，避免不同分数量纲直接混用；
  - 每条 lane 的前 guaranteed_per_lane 固定保留；
  - 再加入全局 RRF 前 global_limit 篇；
  - 最终按“固定保留 + 全局排序”去重合并。
  """
  score_map: Dict[str, float] = {}
  hit_count: Dict[str, int] = {}
  guaranteed_ids: List[str] = []

  for q in queries or []:
    top_ids = get_top_ids(q)
    if not top_ids:
      continue
    if guaranteed_per_lane > 0:
      guaranteed_ids.extend(top_ids[:guaranteed_per_lane])
    for rank_idx, pid in enumerate(top_ids, start=1):
      paper_id = str(pid or "").strip()
      if not paper_id:
        continue
      score_map[paper_id] = score_map.get(paper_id, 0.0) + 1.0 / (RRF_K + rank_idx)
      hit_count[paper_id] = hit_count.get(paper_id, 0) + 1

  ranked = sorted(
    score_map.items(),
    key=lambda item: (
      -item[1],
      -hit_count.get(item[0], 0),
      item[0],
    ),
  )
  global_ids = [pid for pid, _score in ranked]
  if global_limit > 0:
    global_ids = global_ids[:global_limit]
  return _unique_keep_order(list(guaranteed_ids) + list(global_ids))


def iter_batches(
  docs_with_idx: List[Tuple[int, str]],
  query_tokens: int,
  encoder,
) -> List[Tuple[List[int], List[str]]]:
  batches: List[Tuple[List[int], List[str]]] = []
  pos = 0
  while pos < len(docs_with_idx):
    total_tokens = query_tokens
    batch_docs: List[str] = []
    batch_indices: List[int] = []

    while pos < len(docs_with_idx) and len(batch_docs) < BATCH_SIZE:
      orig_idx, doc = docs_with_idx[pos]
      doc_tokens = estimate_tokens(doc, encoder)
      if total_tokens + doc_tokens > TOKEN_SAFETY and batch_docs:
        break
      batch_docs.append(doc)
      batch_indices.append(orig_idx)
      total_tokens += doc_tokens
      pos += 1

    if not batch_docs:
      pos += 1
      continue
    batches.append((batch_indices, batch_docs))
  return batches


def rrf_merge(scores: Dict[int, float], rank_idx: int, orig_idx: int) -> None:
  scores[orig_idx] = scores.get(orig_idx, 0.0) + 1.0 / (RRF_K + rank_idx)


def build_ranked_results(
  score_map: Dict[int, float],
  top_ids: List[str],
  top_n: Optional[int],
) -> List[Dict[str, Any]]:
  if not score_map:
    return []

  sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
  if top_n is not None:
    sorted_items = sorted_items[:top_n]

  score_values = [v for _, v in sorted_items]
  min_score = min(score_values)
  max_score = max(score_values)
  denom = max_score - min_score if max_score > min_score else 1.0

  ranked_for_query: List[Dict[str, Any]] = []
  for idx, raw_score in sorted_items:
    norm_score = (raw_score - min_score) / denom
    paper_id = top_ids[idx]
    ranked_for_query.append(
      {
        'paper_id': paper_id,
        'score': norm_score,
        'star_rating': score_to_stars(norm_score),
      }
    )

  ranked_for_query.sort(key=lambda x: x['score'], reverse=True)
  return ranked_for_query


def rerank_query_with_gateway(
  reranker: GeminiClient,
  q_text: str,
  top_ids: List[str],
  papers_by_id: Dict[str, Dict[str, Any]],
  encoder,
  top_n: Optional[int],
  q_idx: int,
  total_queries: int,
  tag: str,
) -> List[Dict[str, Any]]:
  documents = build_documents(papers_by_id, top_ids)
  docs_with_idx = list(enumerate(documents))
  random.shuffle(docs_with_idx)

  query_tokens = estimate_tokens(q_text, encoder)
  batches = iter_batches(docs_with_idx, query_tokens, encoder)
  log(
    f"[INFO] Query {q_idx}/{total_queries} tag={tag} | candidates={len(top_ids)} "
    f"| batches={len(batches)} | query_tokens≈{query_tokens}"
  )

  rrf_scores: Dict[int, float] = {}
  for batch_idx, (batch_indices, batch_docs) in enumerate(batches, 1):
    log(
      f"[INFO] 发送批次 {batch_idx}/{len(batches)} | docs={len(batch_docs)}"
    )
    response = reranker.rerank(
      query=q_text,
      documents=batch_docs,
      top_n=len(batch_docs),
      model=getattr(reranker, "model", None),
    )
    if isinstance(response, dict) and 'output' in response:
      results = response.get('output', {}).get('results', [])
    else:
      results = response.get('results', [])

    ranked = sorted(
      results or [],
      key=lambda x: x.get('relevance_score', x.get('score', 0.0)),
      reverse=True,
    )
    for rank_idx, item in enumerate(ranked, start=1):
      idx = int(item.get('index', -1))
      if idx < 0 or idx >= len(batch_indices):
        continue
      orig_idx = batch_indices[idx]
      rrf_merge(rrf_scores, rank_idx, orig_idx)

  return build_ranked_results(rrf_scores, top_ids, top_n)


def rerank_query_with_llm_scores(
  reranker: GeminiClient,
  q_text: str,
  top_ids: List[str],
  papers_by_id: Dict[str, Dict[str, Any]],
  encoder,
  top_n: Optional[int],
  q_idx: int,
  total_queries: int,
  tag: str,
) -> List[Dict[str, Any]]:
  documents = build_documents(papers_by_id, top_ids)
  docs_with_idx = list(enumerate(documents))

  query_tokens = estimate_tokens(q_text, encoder)
  batches = iter_batches(docs_with_idx, query_tokens, encoder)
  log(
    f"[INFO] Query {q_idx}/{total_queries} tag={tag} | candidates={len(top_ids)} "
    f"| batches={len(batches)} | query_tokens≈{query_tokens} | strategy=gemini_score"
  )

  score_map: Dict[int, float] = {}
  for batch_idx, (batch_indices, batch_docs) in enumerate(batches, 1):
    log(
      f"[INFO] Gemini 评分批次 {batch_idx}/{len(batches)} | docs={len(batch_docs)}"
    )
    try:
      batch_scores = score_documents_with_llm(reranker, q_text, batch_docs)
    except Exception as exc:
      log(f"[WARN] Gemini 评分失败，回退原始顺序评分：{exc}")
      batch_scores = fallback_batch_scores(batch_docs)

    for item in batch_scores:
      idx = int(item['index'])
      if idx < 0 or idx >= len(batch_indices):
        continue
      orig_idx = batch_indices[idx]
      score_map[orig_idx] = float(item['score'])

  return build_ranked_results(score_map, top_ids, top_n)


def process_file(
  reranker: GeminiClient,
  input_path: str,
  output_path: str,
  top_n: Optional[int],
  rerank_model: str,
) -> None:
  data = load_json(input_path)
  papers_list = data.get("papers") or []
  all_queries = data.get("queries") or []
  if not papers_list or not all_queries:
    log(f"[WARN] 文件 {os.path.basename(input_path)} 中缺少 papers 或 queries，跳过。")
    return

  # 仅使用语义查询（intent_query 或兼容旧的 llm_query）进行 rerank。
  def _is_intent_rerank_query(q: Dict[str, Any]) -> bool:
    q_type = str(q.get("type") or "").strip().lower()
    return q_type in {"intent_query", "llm_query"}

  queries = [q for q in all_queries if _is_intent_rerank_query(q)]
  if not queries:
    log("[WARN] 当前输入中没有可用于 rerank 的意图查询，跳过 rerank。")
    # 保持输出结构一致，避免后续步骤读不到文件
    meta_generated_at = data.get("generated_at") or ""
    data["reranked_at"] = datetime.now(timezone.utc).isoformat()
    data["generated_at"] = meta_generated_at
    save_json(data, output_path)
    return

  papers_by_id = {str(p.get("id")): p for p in papers_list if p.get("id")}
  lane_top_k, guaranteed_per_lane, global_rrf_top = resolve_global_pool_budget(
    len(papers_list),
    len(queries),
  )
  global_candidate_ids = build_global_candidate_ids(
    all_queries,
    guaranteed_per_lane=guaranteed_per_lane,
    global_limit=global_rrf_top,
  )
  data["global_candidate_ids"] = global_candidate_ids
  data["global_pool_lane_top_k"] = lane_top_k
  data["global_pool_limit"] = global_rrf_top
  data["global_pool_guaranteed_per_lane"] = guaranteed_per_lane
  if not global_candidate_ids:
    log("[WARN] 未能从任意 query 中构建统一候选池，跳过 rerank。")
    meta_generated_at = data.get("generated_at") or ""
    data["reranked_at"] = datetime.now(timezone.utc).isoformat()
    data["generated_at"] = meta_generated_at
    save_json(data, output_path)
    return
  encoder = build_token_encoder()
  group_start(f"Step 3 - rerank {os.path.basename(input_path)}")
  log(
    f"[INFO] 开始 rerank：queries={len(queries)}（仅 intent/语义查询），papers={len(papers_list)}，"
    f"global_pool={len(global_candidate_ids)}（lane_top_k={lane_top_k}, "
    f"guaranteed_per_lane={guaranteed_per_lane}, global_top={global_rrf_top}），"
    f"batch_size={BATCH_SIZE}，"
    f"max_chars={MAX_CHARS_PER_DOC}，token_safety={TOKEN_SAFETY}"
  )

  use_gateway_rerank = getattr(
    reranker,
    'supports_rerank',
    lambda: hasattr(reranker, 'rerank'),
  )()
  for q_idx, q in enumerate(queries, start=1):
    q_text = (q.get("rewrite") or q.get("query_text") or "").strip()
    top_ids = list(global_candidate_ids)
    if not q_text or not top_ids:
      continue

    group_start(f"Query {q_idx}/{len(queries)} tag={q.get('tag') or ''}")
    try:
      if use_gateway_rerank:
        ranked_for_query = rerank_query_with_gateway(
          reranker,
          q_text,
          top_ids,
          papers_by_id,
          encoder,
          top_n,
          q_idx,
          len(queries),
          q.get('tag') or '',
        )
      else:
        ranked_for_query = rerank_query_with_llm_scores(
          reranker,
          q_text,
          top_ids,
          papers_by_id,
          encoder,
          top_n,
          q_idx,
          len(queries),
          q.get('tag') or '',
        )
      if not ranked_for_query:
        log("[WARN] 本次 query 未得到有效 rerank 结果，跳过。")
        continue
      q["ranked"] = ranked_for_query
    finally:
      group_end()

  meta_generated_at = data.get("generated_at") or ""
  data["reranked_at"] = datetime.now(timezone.utc).isoformat()
  data["generated_at"] = meta_generated_at

  save_json(data, output_path)
  group_end()


def main() -> None:
  parser = argparse.ArgumentParser(
    description="步骤 3：使用 Gemini 评分或兼容旧网关的 rerank 接口对候选论文做重排序。",
  )
  parser.add_argument(
    "--input",
    type=str,
    default=os.path.join(FILTERED_DIR, f"arxiv_papers_{TODAY_STR}.json"),
    help="筛选结果 JSON 路径。",
  )
  parser.add_argument(
    "--output",
    type=str,
    default=os.path.join(RANKED_DIR, f"arxiv_papers_{TODAY_STR}.json"),
    help="打分后的输出 JSON 路径。",
  )
  parser.add_argument(
    "--top-n",
    type=int,
    default=None,
    help="最终保留的 Top N（默认保留全部候选）。",
  )
  parser.add_argument(
    "--rerank-model",
    type=str,
    default=resolve_model_env("GEMINI_RERANK_MODEL", "BLT_RERANK_MODEL", "gemini-3-flash-preview"),
    help="排序阶段使用的模型名称（默认 gemini-3-flash-preview）。",
  )

  args = parser.parse_args()

  input_path = args.input
  if not os.path.isabs(input_path):
    input_path = os.path.abspath(os.path.join(ROOT_DIR, input_path))

  output_path = args.output
  if not os.path.isabs(output_path):
    output_path = os.path.abspath(os.path.join(ROOT_DIR, output_path))

  if not os.path.exists(input_path):
    log(f"[WARN] 输入文件不存在（今天可能没有新论文）：{input_path}，将跳过 Step 3。")
    return

  api_key = resolve_gemini_api_key()
  if not api_key:
    raise RuntimeError("缺少 GEMINI_API_KEY 环境变量（兼容旧 BLT_API_KEY），无法调用排序模型。")

  reranker = GeminiClient(api_key=api_key, model=args.rerank_model)
  process_file(
    reranker=reranker,
    input_path=input_path,
    output_path=output_path,
    top_n=args.top_n,
    rerank_model=args.rerank_model,
  )


if __name__ == "__main__":
  main()
