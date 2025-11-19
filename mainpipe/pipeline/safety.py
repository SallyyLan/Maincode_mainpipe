"""
Lightweight Safety Filtering
============================

This module runs lightweight rule-based toxicity scoring plus regex-driven
PII detection. It focuses on deterministic heuristics with low-latency
execution so the pipeline can enforce safety without loading large models.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

WORD_PATTERN = re.compile(r"[a-zA-Z']+")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{3,}")
SAFE_SHORT_TEXT_PATTERN = re.compile(r"^[\w\s.,'\"-]{1,40}$")
SAFE_NUMERIC_PATTERN = re.compile(r"^\s*(?:[-+]?[\d.,:%\s]{1,})\s*$")
SAFE_TIMESTAMP_PATTERN = re.compile(
    r"^\s*(?:\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?)\s*$"
)
SAFE_URL_ONLY_PATTERN = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", re.IGNORECASE)
SAFE_FORMAL_TEMPLATE_PATTERN = re.compile(
    r"^(?:dear|sincerely|regards|to whom it may concern)[\s,].{0,200}$",
    re.IGNORECASE,
)
SAFE_CODE_BLOCK_PATTERN = re.compile(r"^```[\s\S]+```$", re.MULTILINE)
SAFE_ALLOWLIST_PHRASES = {
    "thank you",
    "please advise",
    "see attached",
    "looking forward to hearing from you",
}
PROFANITY_TERMS = [
    r"\b(?:fuck|shit|bitch|bastard|asshole|cunt|slut|dick|prick|whore)\b",
    r"\b(?:damn|dumbass|jackass|motherf+er|f+ck|s+hit)\b",
]
PROFANITY_PATTERN = re.compile("|".join(PROFANITY_TERMS), re.IGNORECASE)
OBFUSCATED_PROFANITY_PATTERN = re.compile(r"\b[f5]\W{0,2}u\W{0,2}c\W{0,2}k\b", re.IGNORECASE)
REPEATED_PUNCT_PATTERN = re.compile(r"[!?]{3,}")
ALL_CAPS_WORD_PATTERN = re.compile(r"\b[A-Z]{4,}\b")
ASCII_RATIO_MIN = 0.5
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

DEFAULT_BLOCKLIST = {
    "abuse": 0.8,
    "attack": 0.9,
    "bomb": 1.2,
    "bully": 0.9,
    "death": 0.7,
    "destroy": 0.9,
    "harass": 1.0,
    "hate": 1.1,
    "killer": 1.4,
    "kill": 1.5,
    "murder": 1.7,
    "rape": 2.0,
    "shoot": 1.3,
    "suicide": 1.6,
    "terror": 1.3,
    "threat": 1.2,
    "violence": 1.1,
}

PII_REGEX_PATTERNS = {
    "EMAIL_ADDRESS": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b",
    "PHONE_NUMBER": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]?){13,16}\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}
COMPILED_PII_REGEX = {
    name: re.compile(pattern, re.IGNORECASE) for name, pattern in PII_REGEX_PATTERNS.items()
}

@dataclass
class PrefilterResult:
    tag: str
    reasons: List[str]
    suspect_spans: List[Tuple[int, int, str]]
    metadata: Dict[str, Any]


@dataclass
class TextWindow:
    text: str
    start: int
    end: int
    tag: str
    priority: int
    source: str


RULE_VERSION = "v1"
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b-\u200d\u2060\uFEFF]")
MULTISPACE_PATTERN = re.compile(r"\s+")
REPEATED_SYMBOL_PATTERN = re.compile(r"([^\w\s])\1{2,}")


def _normalize_for_rule_scoring(text: str) -> str:
    """
    Lightweight normalization: strip zero-width chars, collapse whitespace, and
    cap repeated symbols to speed up downstream regex/token checks.
    """

    if not text:
        return ""
    without_zero_width = ZERO_WIDTH_PATTERN.sub("", text)
    collapsed_symbols = REPEATED_SYMBOL_PATTERN.sub(r"\1\1", without_zero_width)
    normalized = MULTISPACE_PATTERN.sub(" ", collapsed_symbols)
    return normalized.strip()


def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return ascii_chars / max(1, len(text))


def _prefilter_text(text: str, config: SafetyConfig) -> PrefilterResult:
    stripped = text.strip()
    if not config.prefilter_enabled:
        return PrefilterResult("unknown", [], [], {"length": len(text)})

    reasons: List[str] = []
    suspect_spans: List[Tuple[int, int, str]] = []
    metadata: Dict[str, Any] = {"length": len(text)}
    severity = 0

    if not stripped:
        return PrefilterResult("safe", ["empty"], suspect_spans, metadata)

    lower = text.lower()
    ascii_ratio_value = _ascii_ratio(text)
    metadata["ascii_ratio"] = round(ascii_ratio_value, 3)

    for pattern, label, weight in [
        (PROFANITY_PATTERN, "profanity", 3),
        (OBFUSCATED_PROFANITY_PATTERN, "obfuscated_profanity", 3),
        (REPEATED_PUNCT_PATTERN, "repeated_punctuation", 2),
        (ALL_CAPS_WORD_PATTERN, "all_caps", 1),
    ]:
        for match in pattern.finditer(text):
            suspect_spans.append((match.start(), match.end(), label))
            severity = max(severity, weight)

    if severity > 0:
        reasons.append("suspect_pattern")

    if ascii_ratio_value < config.prefilter_non_english_ascii_ratio:
        reasons.append("non_english_like")
        severity = max(severity, 1)

    if (
        len(stripped) <= config.prefilter_safe_char_limit
        and SAFE_SHORT_TEXT_PATTERN.match(stripped)
        and ascii_ratio_value >= config.prefilter_safe_ascii_ratio
    ):
        reasons.append("short_neutral")
        severity = min(severity, 1)

    if SAFE_NUMERIC_PATTERN.match(stripped):
        reasons.append("numeric_like")
    if SAFE_TIMESTAMP_PATTERN.match(stripped):
        reasons.append("timestamp_like")
    if SAFE_URL_ONLY_PATTERN.match(stripped):
        reasons.append("url_only")
    if SAFE_FORMAL_TEMPLATE_PATTERN.match(stripped):
        reasons.append("formal_template")
    if SAFE_CODE_BLOCK_PATTERN.match(stripped):
        reasons.append("code_block")
    if any(phrase in lower for phrase in SAFE_ALLOWLIST_PHRASES):
        reasons.append("allowlist_phrase")

    tag = "unknown"
    if severity >= 2:
        tag = "suspect"
    elif severity == 1:
        tag = "unknown"
    else:
        safe_reasons = {
            "short_neutral",
            "numeric_like",
            "timestamp_like",
            "url_only",
            "formal_template",
            "code_block",
            "allowlist_phrase",
            "non_english_like",
        }
        if reasons and all(reason in safe_reasons for reason in reasons):
            tag = "safe"

    metadata["severity"] = severity
    metadata["safe_reasons"] = [r for r in reasons if r != "suspect_pattern"]
    metadata["suspect_spans"] = len(suspect_spans)

    return PrefilterResult(tag, reasons, suspect_spans, metadata)


def _prefilter_to_dict(result: PrefilterResult) -> Dict[str, Any]:
    return {
        "tag": result.tag,
        "reasons": result.reasons,
        "metadata": result.metadata,
        "suspect_spans": [
            {"start": start, "end": end, "label": label} for start, end, label in result.suspect_spans
        ],
    }


def _select_text_windows(text: str, config: SafetyConfig, prefilter: PrefilterResult) -> List[TextWindow]:
    if not text:
        return []

    window_chars = max(128, int(config.toxicity_window_chars or 256))
    char_budget = max(window_chars, int(config.toxicity_char_cap or window_chars))
    severity = int(prefilter.metadata.get("severity", 0))
    max_windows = math.ceil(char_budget / window_chars)
    if severity >= 3:
        limit = min(config.toxicity_suspect_windows, max_windows)
    elif severity == 2:
        limit = min(max(2, config.toxicity_max_windows), max_windows)
    else:
        limit = min(config.toxicity_max_windows, max_windows)

    text_length = len(text)
    windows: List[TextWindow] = []
    seen: set[Tuple[int, int]] = set()

    def register(start: int, end: int, tag: str, priority: int, source: str) -> None:
        if len(windows) >= config.toxicity_suspect_windows:
            return
        start = max(0, start)
        end = min(text_length, max(start + 1, end))
        if end - start > window_chars:
            end = start + window_chars
        key = (start, end)
        if key in seen:
            return
        seen.add(key)
        windows.append(TextWindow(text[start:end], start, end, tag, priority, source))

    register(0, window_chars, "head", 2, "head")
    register(text_length - window_chars, text_length, "tail", 2, "tail")

    for span_start, span_end, label in prefilter.suspect_spans[: config.toxicity_suspect_windows]:
        center = (span_start + span_end) // 2
        win_start = max(0, center - window_chars // 2 - config.toxicity_window_padding)
        register(win_start, win_start + window_chars, f"match_{label}", 0, f"match:{label}")

    if severity >= 3 and len(windows) < limit:
        stride = int(window_chars * (1 - min(max(config.toxicity_window_stride, 0.1), 0.9)))
        stride = max(64, stride)
        cursor = 0
        while cursor < text_length and len(windows) < limit:
            register(cursor, cursor + window_chars, "slide", 1, "slide")
            cursor += stride

    if len(windows) < limit:
        mid_start = max(0, (text_length // 2) - (window_chars // 2))
        register(mid_start, mid_start + window_chars, "middle", 3, "middle")

    windows.sort(key=lambda w: (w.priority, w.start))
    selected: List[TextWindow] = []
    remaining_budget = char_budget
    remaining_limit = limit

    for window in windows:
        if remaining_budget <= 0 or remaining_limit <= 0:
            break
        snippet = window.text
        if len(snippet) > remaining_budget:
            snippet = snippet[:remaining_budget]
            end = window.start + len(snippet)
            selected.append(
                TextWindow(snippet, window.start, end, window.tag, window.priority, window.source)
            )
            break
        selected.append(window)
        remaining_budget -= len(snippet)
        remaining_limit -= 1

    return selected


def _normalize_blocklist(value: Optional[Any]) -> Dict[str, float]:
    if value is None:
        return dict(DEFAULT_BLOCKLIST)
    if isinstance(value, dict):
        normalized = {}
        for key, weight in value.items():
            if not key:
                continue
            try:
                normalized[str(key).lower()] = float(weight)
            except (TypeError, ValueError):
                normalized[str(key).lower()] = 1.0
        return normalized
    if isinstance(value, (list, tuple, set)):
        return {str(token).lower(): 1.0 for token in value if token}
    raise TypeError("Blocklist must be a dict or iterable of strings")


def _normalize_pii_entities(value: Optional[Iterable[str]]) -> List[str]:
    if not value:
        return list(COMPILED_PII_REGEX.keys())
    return [str(entity).upper() for entity in value if str(entity).upper() in COMPILED_PII_REGEX]


@dataclass
class SafetyConfig:
    """
    Tunable parameters for the lightweight safety checks.
    """

    enabled: bool = True
    toxicity_threshold: float = 1.5
    blocklist: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BLOCKLIST))
    max_chars_scanned: int = 10000  # PII detector still processes up to this limit
    toxicity_char_cap: int = 768  # Lower cap: rule heuristics detect local toxicity early
    toxicity_window_chars: int = 512
    toxicity_window_stride: float = 0.5  # 50% overlap for sliding window when needed
    toxicity_window_padding: int = 64
    toxicity_max_windows: int = 3
    toxicity_suspect_windows: int = 5
    prefilter_enabled: bool = True
    prefilter_safe_char_limit: int = 20
    prefilter_safe_ascii_ratio: float = 0.75
    prefilter_safe_skip_toxicity: bool = True
    prefilter_non_english_ascii_ratio: float = 0.35
    uppercase_ratio_limit: float = 0.65
    symbol_ratio_limit: float = 0.45
    max_repeated_chars: int = 4
    uppercase_penalty: float = 0.5
    symbol_penalty: float = 0.4
    repeat_penalty: float = 0.4
    repeated_punct_penalty: float = 0.3
    obfuscated_profanity_penalty: float = 0.6
    min_chars_for_uppercase_check: int = 20
    enable_pii: bool = True
    pii_entities: List[str] = field(default_factory=lambda: list(COMPILED_PII_REGEX.keys()))
    pii_max_matches: int = 2

    @classmethod
    def from_overrides(
        cls,
        base: Optional["SafetyConfig"] = None,
        **overrides: Any,
    ) -> "SafetyConfig":
        if base is None:
            data: Dict[str, Any] = {}
        elif isinstance(base, cls):
            data = asdict(base)
        elif isinstance(base, dict):
            data = dict(base)
        else:
            raise TypeError("base must be SafetyConfig, dict, or None")

        for key, value in overrides.items():
            if value is not None:
                data[key] = value

        data["blocklist"] = _normalize_blocklist(data.get("blocklist"))
        data["pii_entities"] = _normalize_pii_entities(data.get("pii_entities"))

        return cls(**data)


def empty_toxicity_result() -> Dict[str, Any]:
    return {
        "is_toxic": False,
        "score": 0.0,
        "flagged_tokens": [],
        "signals": {"rule_version": RULE_VERSION},
        "char_span": 0,
    }


def empty_pii_result() -> Dict[str, Any]:
    return {
        "has_pii": False,
        "pii_types": [],
        "pii_count": 0,
    }


def _truncate_text(text: str, limit: int) -> str:
    if limit and limit > 0:
        return text[:limit]
    return text


def _has_repeated_sequence(text: str, threshold: int) -> bool:
    if threshold <= 1:
        return True
    streak = 1
    for prev, curr in zip(text, text[1:]):
        if curr == prev:
            streak += 1
            if streak >= threshold:
                return True
        else:
            streak = 1
    return False


def _score_toxicity_rule_based(
    text: str,
    config: SafetyConfig,
    *,
    fallback_reason: Optional[str] = None,
    already_truncated: bool = False,
) -> Dict[str, Any]:
    if not text:
        result = empty_toxicity_result()
        if fallback_reason:
            result["signals"]["fallback"] = fallback_reason
        return result

    truncated = text if already_truncated else _truncate_text(text, config.toxicity_char_cap)
    normalized = _normalize_for_rule_scoring(truncated)
    lowered = normalized.lower()
    tokens = WORD_PATTERN.findall(lowered)
    flagged_tokens: List[str] = []
    score = 0.0

    blocklist_hits = 0
    for token in tokens:
        weight = config.blocklist.get(token)
        if weight:
            score += weight
            flagged_tokens.append(token)
            blocklist_hits += 1

    letters = sum(1 for ch in truncated if ch.isalpha())
    uppercase_letters = sum(1 for ch in truncated if ch.isupper())
    uppercase_ratio = (uppercase_letters / letters) if letters else 0.0
    symbol_chars = sum(1 for ch in truncated if not ch.isalnum() and not ch.isspace())
    symbol_ratio = symbol_chars / max(1, len(truncated))
    repeated_chars = (
        _has_repeated_sequence(truncated, config.max_repeated_chars)
        or bool(REPEATED_CHAR_PATTERN.search(truncated))
    )
    repeated_punct_matches = REPEATED_PUNCT_PATTERN.findall(truncated)
    obfuscated_hits = OBFUSCATED_PROFANITY_PATTERN.findall(lowered)

    signals: Dict[str, Any] = {
        "rule_version": RULE_VERSION,
        "blocklist_hits": blocklist_hits,
        "uppercase_ratio": round(uppercase_ratio, 3),
        "symbol_ratio": round(symbol_ratio, 3),
    }

    if (
        len(truncated) >= config.min_chars_for_uppercase_check
        and uppercase_ratio > config.uppercase_ratio_limit
    ):
        score += config.uppercase_penalty
        signals["uppercase_penalty_applied"] = True
    if symbol_ratio > config.symbol_ratio_limit:
        score += config.symbol_penalty
        signals["symbol_penalty_applied"] = True
    if repeated_chars:
        score += config.repeat_penalty
        signals["repeated_sequence"] = True
    if repeated_punct_matches:
        score += config.repeated_punct_penalty
        signals["repeated_punctuation"] = len(repeated_punct_matches)
    if obfuscated_hits:
        score += config.obfuscated_profanity_penalty
        signals["obfuscated_matches"] = len(obfuscated_hits)
        signals["obfuscated_examples"] = obfuscated_hits[:3]

    if fallback_reason:
        signals["fallback"] = fallback_reason

    result = {
        "is_toxic": score >= config.toxicity_threshold,
        "score": round(score, 3),
        "flagged_tokens": flagged_tokens,
        "signals": signals,
        "char_span": len(truncated),
    }
    return result


def _score_toxicity(
    text: str,
    config: SafetyConfig,
    prefilter: PrefilterResult,
) -> Tuple[Dict[str, Any], bool, Dict[str, int]]:
    if not text:
        return empty_toxicity_result(), False, {"cache_hits": 0, "cache_misses": 0, "windows_scored": 0}

    windows = _select_text_windows(text, config, prefilter)
    if not windows:
        truncated = _truncate_text(text, config.toxicity_char_cap)
        stitched = truncated
    else:
        stitched_parts: List[str] = []
        remaining_budget = config.toxicity_char_cap
        for window in windows:
            if remaining_budget <= 0:
                break
            snippet = window.text[:remaining_budget]
            if snippet:
                stitched_parts.append(snippet)
                remaining_budget -= len(snippet)
        stitched = "\n".join(stitched_parts) if stitched_parts else _truncate_text(text, config.toxicity_char_cap)

    stitched = stitched[: config.toxicity_char_cap]
    result = _score_toxicity_rule_based(
        stitched,
        config,
        fallback_reason=None,
        already_truncated=True,
    )
    result["char_span"] = len(stitched)
    result.setdefault("signals", {})["window_count"] = len(windows)
    result["signals"]["processing_mode"] = "rule_based"

    cache_stats = {"cache_hits": 0, "cache_misses": 0, "windows_scored": len(windows)}
    return result, False, cache_stats


def _detect_pii(text: str, config: SafetyConfig) -> Dict[str, Any]:
    if not config.enable_pii:
        return empty_pii_result()

    truncated = _truncate_text(text, config.max_chars_scanned)
    hits: List[str] = []
    match_budget = config.pii_max_matches

    for entity in config.pii_entities:
        pattern = COMPILED_PII_REGEX.get(entity)
        if pattern is None:
            continue
        for _match in pattern.finditer(truncated):
            hits.append(entity)
            match_budget -= 1
            if match_budget <= 0:
                break
        if match_budget <= 0:
            break

    if not hits:
        return empty_pii_result()

    return {
        "has_pii": True,
        "pii_types": hits,
        "pii_count": len(hits),
    }


def filter_samples(
    samples: List[Dict[str, Any]],
    config: Optional[SafetyConfig] = None,
    **overrides: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run toxicity + PII checks over the provided samples.
    """

    cfg = SafetyConfig.from_overrides(config, **overrides)
    total = len(samples)

    if not cfg.enabled:
        logger.info("Safety filtering disabled; adding pii/toxicity fields to %d samples without filtering.", total)
        for sample in samples:
            text = sample.get("text", "")
            prefilter = _prefilter_text(text, cfg)
            sample["prefilter"] = _prefilter_to_dict(prefilter)
            if text:
                tox_result, _, _ = _score_toxicity(text, cfg, prefilter)
                sample["toxicity"] = tox_result
                sample["pii"] = _detect_pii(text, cfg)
            else:
                toxicity = empty_toxicity_result()
                toxicity["signals"]["prefilter_tag"] = prefilter.tag
                sample["toxicity"] = toxicity
                sample["pii"] = empty_pii_result()
            sample["drop_reason"] = None

        metrics = {
            "enabled": False,
            "total_samples": total,
            "kept_samples": total,
            "toxic_dropped": 0,
            "pii_dropped": 0,
            "processing_mode": "rule_based",
        }
        return samples, metrics

    kept: List[Dict[str, Any]] = []
    toxic_dropped = 0
    pii_dropped = 0
    empty_texts = 0
    total_chars = 0
    total_score = 0.0
    flagged_token_counts: Counter[str] = Counter()
    pii_type_counts: Counter[str] = Counter()
    prefilter_counts: Counter[str] = Counter()
    detoxify_cache_hits = 0
    detoxify_cache_misses = 0
    detoxify_windows_total = 0
    detoxify_scored_samples = 0

    for sample in samples:
        text = sample.get("text", "")
        prefilter = _prefilter_text(text, cfg)
        sample["prefilter"] = _prefilter_to_dict(prefilter)
        prefilter_counts[prefilter.tag] += 1

        if not text:
            empty_texts += 1
            toxicity = empty_toxicity_result()
            toxicity["signals"]["prefilter_tag"] = prefilter.tag
            toxicity["signals"]["processing_mode"] = "rule_based"
            sample["toxicity"] = toxicity
            sample["pii"] = empty_pii_result()
            sample["drop_reason"] = None
            kept.append(sample)
            continue

        if prefilter.tag == "safe" and cfg.prefilter_safe_skip_toxicity:
            toxicity = empty_toxicity_result()
            toxicity["signals"]["prefilter_tag"] = "safe"
            toxicity["signals"]["processing_mode"] = "rule_based"
            toxicity["signals"]["skip_reason"] = "prefilter_safe"
            toxicity["signals"]["safe_reasons"] = prefilter.metadata.get("safe_reasons", [])
            sample["toxicity"] = toxicity
            pii_result = _detect_pii(text, cfg)
            sample["pii"] = pii_result
            logger.debug(
                "Prefilter safe skip: reasons=%s",
                toxicity["signals"]["safe_reasons"],
            )
            if pii_result.get("has_pii"):
                pii_dropped += 1
                sample["drop_reason"] = "contains_pii"
                pii_type_counts.update(pii_result.get("pii_types", []))
                continue
            sample["drop_reason"] = None
            kept.append(sample)
            continue

        tox_result, _, cache_stats = _score_toxicity(text, cfg, prefilter)
        tox_result.setdefault("signals", {})["prefilter_tag"] = prefilter.tag
        sample["toxicity"] = tox_result
        total_chars += tox_result["char_span"]
        total_score += tox_result["score"]
        flagged_token_counts.update(tox_result["flagged_tokens"])

        detoxify_cache_hits += cache_stats.get("cache_hits", 0)
        detoxify_cache_misses += cache_stats.get("cache_misses", 0)
        detoxify_windows_total += cache_stats.get("windows_scored", 0)
        detoxify_scored_samples += 1

        if tox_result["is_toxic"]:
            toxic_dropped += 1
            sample["drop_reason"] = "toxic"
            sample["pii"] = empty_pii_result()
            continue

        pii_result = _detect_pii(text, cfg)
        sample["pii"] = pii_result
        if pii_result.get("has_pii"):
            pii_dropped += 1
            sample["drop_reason"] = "contains_pii"
            pii_type_counts.update(pii_result.get("pii_types", []))
            continue
        
        sample["drop_reason"] = None
        kept.append(sample)

    metrics = {
        "enabled": True,
        "total_samples": total,  # ALL samples entering safety stage
        "kept_samples": len(kept),  # Samples that passed safety filtering
        "toxic_dropped": toxic_dropped,  # Number of SAMPLES dropped (not occurrences)
        "pii_dropped": pii_dropped,  # Number of SAMPLES dropped (not occurrences)
        "empty_texts": empty_texts,
        # avg_chars_scanned: Average chars scanned across ALL samples entering safety (not just kept)
        # This can be < avg_char_length of kept samples because:
        # 1. max_chars_scanned limit (2000) truncates long texts
        # 2. Calculated on all samples, not just kept ones
        "avg_chars_scanned": (total_chars / max(1, total - empty_texts)),
        # avg_toxicity_score: Average toxicity score across ALL samples entering safety (not just kept)
        "avg_toxicity_score": (total_score / max(1, total - empty_texts)),
        "flagged_token_counts": dict(flagged_token_counts.most_common(20)),
        # pii_type_counts: Count of PII type OCCURRENCES (not samples)
        # A single sample can have multiple PII types, so sum(pii_type_counts) >= pii_dropped
        # Example: A sample with both EMAIL and PHONE counts as 1 in pii_dropped but 2 in pii_type_counts
        "pii_type_counts": dict(pii_type_counts),
        "prefilter_counts": dict(prefilter_counts),
        "detoxify_cache_hits": detoxify_cache_hits,
        "detoxify_cache_misses": detoxify_cache_misses,
        "avg_windows_scored": (
            detoxify_windows_total / max(1, detoxify_scored_samples)
            if detoxify_scored_samples
            else 0.0
        ),
        "processing_mode": "rule_based",
        "config_snapshot": {
            "rule_version": RULE_VERSION,
            "toxicity_threshold": cfg.toxicity_threshold,
            "max_chars_scanned": cfg.max_chars_scanned,
            "toxicity_char_cap": cfg.toxicity_char_cap,
            "toxicity_window_chars": cfg.toxicity_window_chars,
            "toxicity_threshold_mode": "absolute_weight_sum",
            "toxicity_char_cap_note": "Lower cap ensures quick rule evaluation; toxic spans trigger locally.",
            "prefilter_enabled": cfg.prefilter_enabled,
            "prefilter_safe_skip_toxicity": cfg.prefilter_safe_skip_toxicity,
            "uppercase_ratio_limit": cfg.uppercase_ratio_limit,
            "symbol_ratio_limit": cfg.symbol_ratio_limit,
            "max_repeated_chars": cfg.max_repeated_chars,
            "uppercase_penalty": cfg.uppercase_penalty,
            "symbol_penalty": cfg.symbol_penalty,
            "repeat_penalty": cfg.repeat_penalty,
            "repeated_punct_penalty": cfg.repeated_punct_penalty,
            "obfuscated_profanity_penalty": cfg.obfuscated_profanity_penalty,
            "blocklist_size": len(cfg.blocklist),
            "pii_entities": cfg.pii_entities,
        },
    }

    logger.info(
        "Safety filtering kept %d/%d samples (toxic=%d, pii=%d, empty=%d)",
        len(kept),
        total,
        toxic_dropped,
        pii_dropped,
        empty_texts,
    )

    return kept, metrics


__all__ = ["SafetyConfig", "filter_samples"]


