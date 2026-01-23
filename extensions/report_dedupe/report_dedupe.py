# coding=utf-8
"""
Report Deduplication Plugin

This plugin merges similar news titles from different platforms, reducing duplicates
and presenting cleaner, more focused results.
"""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

# OllamaClient removed - now using main AI client via extensions.ai_client


# Regex to extract alphanumeric words (for English/numbers)
ALNUM_RE = re.compile(r"[a-zA-Z0-9]+")
# Regex to match Chinese characters (will be split individually)
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")

# Module-level cache for deduplication results (within single run only)
_dedupe_cache = {}


def clear_dedupe_cache() -> None:
    """Clear the deduplication cache. Should be called at the start of each run."""
    global _dedupe_cache
    _dedupe_cache.clear()
    logger.debug("[report_dedupe] Cache cleared")


def _generate_cache_key(report_data: Dict[str, Any]) -> str:
    """
    Generate a unique cache key from report data using MD5 hash.

    Args:
        report_data: Report data dictionary

    Returns:
        MD5 hash string of the serialized report data
    """
    data_str = json.dumps(report_data, sort_keys=True).encode()
    return hashlib.md5(data_str).hexdigest()


def transform_report_data(
    report_data: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Transform report data by deduplicating similar titles.

    Args:
        report_data: Report data dictionary
        config: Plugin configuration (runtime format with uppercase keys)

    Returns:
        Transformed report data with deduplicated titles
    """
    if not config or not config.get("enabled"):
        return report_data

    stats = report_data.get("stats", [])
    new_titles = report_data.get("new_titles", [])

    # Early return only if both stats and new_titles are empty
    if not stats and not new_titles:
        return report_data

    # Default to "ollama" strategy for better deduplication accuracy
    strategy = str(config.get("strategy", "ollama")).lower()
    similarity_config = config.get("similarity", {})
    merge_config = config.get("merge", {})
    threshold = _coerce_float(similarity_config.get("threshold", 0.85), 0.85)
    candidate_floor_multiplier = _coerce_float(
        similarity_config.get("candidate_floor_multiplier", 0.35), 0.35
    )
    max_ai_checks = _coerce_int(similarity_config.get("max_ai_checks", 50), 50)
    max_items_per_group = _coerce_int(merge_config.get("max_items_per_group", 10), 10)

    logger.debug(
        "[report_dedupe] Config: strategy={}, threshold={}, candidate_floor_multiplier={}, max_ai_checks={}",
        strategy,
        threshold,
        candidate_floor_multiplier,
        max_ai_checks,
    )

    client = None
    client_model_override = None
    if strategy in ("ollama", "auto"):
        ollama_config = config.get("ollama", {})
        prompt_file = config.get("prompt_file")

        # 优先使用主 AI 客户端（通过 context 传入）
        ai_client = config.get("ai_client")
        if ai_client:
            client = ai_client
            # 主 AI 客户端已配置好模型，这里使用 None
            client_model_override = None
            logger.info(
                "[report_dedupe] Using main AI client with Ollama model"
            )
        else:
            # No AI client available, use heuristic-only mode
            logger.warning(
                "[report_dedupe] AI client not available, falling back to heuristic-only mode"
            )

    ai_state = {
        "remaining": max_ai_checks,
        "client": client,
        "client_model_override": client_model_override,
        "candidate_floor_multiplier": candidate_floor_multiplier,
        "threshold": threshold,
    }

    # Count original titles before deduplication
    original_count = sum(len(stat.get("titles", [])) for stat in stats)
    logger.debug(
        "[report_dedupe] Processing {} stats with {} total titles",
        len(stats),
        original_count,
    )

    # Check cache before deduplication
    cache_key = _generate_cache_key(report_data)
    if cache_key in _dedupe_cache:
        logger.debug("[report_dedupe] Using cached deduplication result")
        return _dedupe_cache[cache_key]

    updated_stats = []
    for stat in stats:
        titles = stat.get("titles", [])
        merged_titles = _dedupe_titles(
            titles,
            threshold,
            merge_config,
            max_items_per_group,
            ai_state,
            strategy,
        )
        stat["titles"] = merged_titles
        stat["count"] = len(merged_titles)
        updated_stats.append(stat)

    total_count = sum(len(stat.get("titles", [])) for stat in updated_stats)
    for stat in updated_stats:
        if total_count > 0:
            stat["percentage"] = round(stat.get("count", 0) / total_count * 100, 1)
        else:
            stat["percentage"] = 0

    # Log deduplication results
    merged_count = total_count
    logger.info(
        "[report_dedupe] Deduplicated {} -> {} titles ({} removed)",
        original_count,
        merged_count,
        original_count - merged_count,
    )

    report_data["stats"] = updated_stats

    # Also deduplicate new_titles section (cross-source deduplication)
    # Note: new_titles is already fetched at the start of the function
    if new_titles:
        report_data["new_titles"] = _dedupe_new_titles(
            new_titles,
            threshold,
            merge_config,
            max_items_per_group,
            ai_state,
            strategy,
        )

    # Cache the result
    _dedupe_cache[cache_key] = report_data

    return report_data


def _dedupe_titles(
    titles: Sequence[Dict[str, Any]],
    threshold: float,
    merge_config: Dict[str, Any],
    max_items_per_group: int,
    ai_state: Dict[str, Any],
    strategy: str,
) -> List[Dict[str, Any]]:
    # Log titles being processed
    if len(titles) > 1:
        logger.debug("[report_dedupe] Processing {} titles in group", len(titles))
        for i, t in enumerate(titles[:5]):  # Log first 5 titles
            logger.debug(
                "[report_dedupe]   {}. {} | '{}'",
                i + 1,
                t.get("source_name", "?"),
                t.get("title", "")[:40],
            )

    groups: List[List[Dict[str, Any]]] = []
    for item in titles:
        placed = False
        for group in groups:
            if max_items_per_group > 0 and len(group) >= max_items_per_group:
                continue
            if _is_same_title(item, group[0], threshold, ai_state, strategy):
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])

    # Log groups with more than one item (actual merges)
    multi_groups = [g for g in groups if len(g) > 1]
    if multi_groups:
        for g in multi_groups:
            sources = [i.get("source_name", "?") for i in g]
            logger.debug(
                "[report_dedupe] Merged {} items from: {}", len(g), ", ".join(sources)
            )

    merged = [_merge_group(group, merge_config) for group in groups if group]
    merged.sort(key=_title_rank)
    return merged


def _dedupe_new_titles(
    new_titles: List[Dict[str, Any]],
    threshold: float,
    merge_config: Dict[str, Any],
    max_items_per_group: int,
    ai_state: Dict[str, Any],
    strategy: str,
) -> List[Dict[str, Any]]:
    """
    Deduplicate new_titles across all sources.

    The new_titles structure is:
    [
        {"source_id": "...", "source_name": "...", "titles": [...]},
        ...
    ]

    This function flattens all titles, deduplicates them using the same algorithm
    as _dedupe_titles, then returns a single unified list.
    """
    # Flatten all titles from all sources
    all_titles: List[Dict[str, Any]] = []
    for source_data in new_titles:
        if not isinstance(source_data, dict):
            continue
        titles_list = source_data.get("titles", [])
        for title_info in titles_list:
            if isinstance(title_info, dict):
                all_titles.append(title_info)

    if not all_titles:
        return new_titles

    original_count = len(all_titles)
    logger.debug(
        "[report_dedupe] Processing {} new_titles for cross-source deduplication",
        original_count,
    )

    # Use the same deduplication algorithm
    groups: List[List[Dict[str, Any]]] = []
    for item in all_titles:
        placed = False
        for group in groups:
            if max_items_per_group > 0 and len(group) >= max_items_per_group:
                continue
            if _is_same_title(item, group[0], threshold, ai_state, strategy):
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])

    # Log cross-source merges
    multi_groups = [g for g in groups if len(g) > 1]
    if multi_groups:
        for g in multi_groups:
            sources = [i.get("source_name", "?") for i in g]
            titles = [i.get("title", "?")[:30] for i in g]
            logger.info(
                "[report_dedupe] New titles merged {} items from {}: {}",
                len(g),
                ", ".join(set(sources)),
                titles[0],
            )

    # Merge groups and create unified result
    merged_titles = [_merge_group(group, merge_config) for group in groups if group]
    merged_titles.sort(key=_title_rank)

    merged_count = len(merged_titles)
    if original_count != merged_count:
        logger.info(
            "[report_dedupe] New titles deduplicated {} -> {} ({} removed)",
            original_count,
            merged_count,
            original_count - merged_count,
        )

    # Return as a single source group with all merged titles
    # This maintains compatibility with the expected structure
    if merged_titles:
        return [
            {
                "source_id": "merged",
                "source_name": "综合来源",
                "titles": merged_titles,
            }
        ]

    return []


def _is_same_title(
    first: Dict[str, Any],
    second: Dict[str, Any],
    threshold: float,
    ai_state: Dict[str, Any],
    strategy: str,
) -> bool:
    title_a = str(first.get("title", ""))
    title_b = str(second.get("title", ""))
    if not title_a or not title_b:
        return False

    normalized_a = _normalize_title(title_a)
    normalized_b = _normalize_title(title_b)
    if normalized_a and normalized_a == normalized_b:
        logger.debug(
            "[report_dedupe] Exact match: '{}' == '{}'", title_a[:30], title_b[:30]
        )
        return True

    tokens_a = _tokenize(normalized_a)
    tokens_b = _tokenize(normalized_b)
    jaccard = _jaccard_similarity(tokens_a, tokens_b)

    if jaccard >= threshold:
        logger.debug(
            "[report_dedupe] Jaccard match ({:.3f}): '{}' ~ '{}'",
            jaccard,
            title_a[:30],
            title_b[:30],
        )
        return True

    if strategy == "heuristic":
        return False

    client = ai_state.get("client")
    if client is None or ai_state.get("remaining", 0) <= 0:
        return False

    # Use configurable floor: lower for Chinese content with synonyms
    floor_multiplier = ai_state.get("candidate_floor_multiplier", 0.35)
    candidate_floor = max(0.2, threshold * floor_multiplier)
    if jaccard < candidate_floor:
        return False

    # Use AI for borderline cases
    logger.debug(
        "[report_dedupe] AI check (jaccard={:.3f}): '{}' vs '{}'",
        jaccard,
        title_a[:30],
        title_b[:30],
    )
    ai_state["remaining"] -= 1

    # 调用 AI 判断相似度
    result = _call_ai_for_similarity(
        client,
        title_a,
        title_b,
        source_a=first.get("source_name", ""),
        source_b=second.get("source_name", ""),
        time_a=first.get("time_display", ""),
        time_b=second.get("time_display", ""),
        count_a=str(first.get("count", 1)),
        count_b=str(second.get("count", 1)),
    )
    if not result:
        logger.debug("[report_dedupe] AI returned no result")
        return False

    same = bool(result.get("same"))
    try:
        confidence = float(result.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0

    if same and confidence >= threshold:
        logger.debug(
            "[report_dedupe] AI match (confidence={:.2f}): '{}' ~ '{}'",
            confidence,
            title_a[:30],
            title_b[:30],
        )
        return True
    else:
        logger.debug(
            "[report_dedupe] AI no match (same={}, conf={:.2f})", same, confidence
        )
        return False


def _call_ai_for_similarity(
    client,
    title_a: str,
    title_b: str,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    调用 AI 判断两个标题是否描述同一事件

    Args:
        client: AIClient 实例
        title_a: 标题 A
        title_b: 标题 B
        **kwargs: 额外参数（source_a, source_b, time_a, time_b, count_a, count_b）

    Returns:
        {"same": bool, "confidence": float} 或 None
    """
    source_a = kwargs.get("source_a", "")
    source_b = kwargs.get("source_b", "")
    time_a = kwargs.get("time_a", "")
    time_b = kwargs.get("time_b", "")
    count_a = kwargs.get("count_a", "1")
    count_b = kwargs.get("count_b", "1")

    # 构建提示词
    prompt = f"""判断以下两个新闻标题是否描述同一新闻事件，只返回JSON对象，字段为same(布尔)和confidence(0到1的小数)。

标题A（{source_a}，出现{count_a}次）：{title_a}
标题B（{source_b}，出现{count_b}次）：{title_b}

时间信息：
- 标题A：{time_a}
- 标题B：{time_b}

请返回JSON格式的判断结果。"""

    try:
        # 使用 AIClient 的 chat 方法
        messages = [{"role": "user", "content": prompt}]
        response = client.chat(messages)

        if not response:
            return None

        # 尝试解析 JSON
        import re
        try:
            # 直接解析
            result = json.loads(response)
            if isinstance(result, dict) and "same" in result:
                return result
        except json.JSONDecodeError:
            pass

        # 尝试从响应中提取 JSON
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            try:
                result = json.loads(json_match.group())
                if isinstance(result, dict) and "same" in result:
                    return result
            except json.JSONDecodeError:
                pass

        return None
    except Exception as e:
        logger.error("[report_dedupe] AI similarity check error: {}", e)
        return None


def _merge_group(
    group: Sequence[Dict[str, Any]], merge_config: Dict[str, Any]
) -> Dict[str, Any]:
    source_separator = merge_config.get("source_separator", " / ")
    count_strategy = merge_config.get("count_strategy", "sum")

    canonical = min(group, key=_title_rank)
    merged = dict(canonical)

    sources = []
    seen_sources = set()
    for item in group:
        name = item.get("source_name")
        if name and name not in seen_sources:
            seen_sources.add(name)
            sources.append(name)
    if sources:
        merged["source_name"] = source_separator.join(sources)

    counts = [_coerce_int(item.get("count", 1), 1) for item in group]
    if count_strategy == "max":
        merged["count"] = max(counts)
    else:
        merged["count"] = sum(counts)

    merged["ranks"] = _merge_ranks(group)
    merged["is_new"] = any(item.get("is_new") for item in group)

    if not merged.get("time_display"):
        for item in group:
            if item.get("time_display"):
                merged["time_display"] = item["time_display"]
                break

    # Collect all unique URLs with their source names for multi-platform links
    urls = []
    seen_urls = set()
    for item in group:
        url = item.get("url") or item.get("mobile_url")
        source = item.get("source_name", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            urls.append({"url": url, "source": source})

    logger.debug(
        "[report_dedupe] _merge_group: collected {} URLs for merged item: {}",
        len(urls),
        merged.get("title", "")[:30],
    )

    # Store all URLs for multi-platform display
    if urls:
        merged["urls"] = urls
        # Keep primary URL for backward compatibility
        merged["url"] = urls[0]["url"]
        # Find first mobile_url if available
        for item in group:
            if item.get("mobile_url"):
                merged["mobile_url"] = item["mobile_url"]
                break

    return merged


def _merge_ranks(group: Sequence[Dict[str, Any]]) -> List[int]:
    ranks: List[int] = []
    seen = set()
    for item in group:
        for rank in item.get("ranks", []) or []:
            if rank not in seen:
                seen.add(rank)
                ranks.append(rank)
    if not ranks:
        return [99]
    return sorted(ranks)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _title_rank(item: Dict[str, Any]) -> int:
    ranks = item.get("ranks") or []
    if ranks:
        return min(ranks)
    return 999


def _normalize_title(title: str) -> str:
    text = title.strip().lower()
    text = re.sub(r"[\[【].*?[\]】]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(title: str) -> List[str]:
    """
    Tokenize title for similarity comparison.
    Chinese characters are split individually, alphanumeric words are kept together.
    """
    if not title:
        return []

    tokens = []
    # Add individual Chinese characters as tokens
    tokens.extend(CHINESE_RE.findall(title))
    # Add alphanumeric words as tokens (for English words, numbers, etc.)
    tokens.extend(ALNUM_RE.findall(title.lower()))
    return tokens


def _jaccard_similarity(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)
