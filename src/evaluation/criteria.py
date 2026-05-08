"""
Multimodal attack success criteria.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "and", "or",
    "photo", "picture", "image", "scene", "photo", "photoes", "photoes", "main",
    "object", "detailed", "describe", "description", "caption", "this", "that",
    "is", "are", "be", "by", "from", "as", "it", "its", "there", "you", "i", "we",
}


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def tokenize_content(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def overlap_score(a: str, b: str) -> float:
    return jaccard_similarity(tokenize_content(a), tokenize_content(b))


def _join_reason(parts: List[str]) -> str:
    return "；".join(part for part in parts if part)


def clip_attack_decision(
    metrics: Dict[str, float],
    targeted: bool = True,
    source_margin: float = 0.01,
    target_margin: float = 0.01,
) -> Dict[str, float | bool | str]:
    """Deterministic CLIP attack decision based on similarity deltas."""
    orig_source = float(metrics.get("orig_source_similarity", 0.0))
    adv_source = float(metrics.get("adv_source_similarity", 0.0))
    orig_target = float(metrics.get("orig_target_similarity", orig_source))
    adv_target = float(metrics.get("adv_target_similarity", metrics.get("adv_source_similarity", 0.0)))
    source_drop = orig_source - adv_source
    target_gain = adv_target - orig_target

    if targeted:
        success = source_drop >= source_margin and target_gain >= target_margin and adv_target >= adv_source
        reason = _join_reason([
            f"源相似度下降 {source_drop:.4f}",
            f"目标相似度提升 {target_gain:.4f}",
        ])
    else:
        success = source_drop >= source_margin
        reason = _join_reason([
            f"源相似度下降 {source_drop:.4f}",
        ])

    return {
        "task": "clip",
        "targeted": targeted,
        "attack_success": bool(success),
        "source_similarity_drop": source_drop,
        "target_similarity_gain": target_gain,
        "rule": "targeted: source_drop>=source_margin and target_gain>=target_margin and adv_target>=adv_source; untargeted: source_drop>=source_margin",
        "reason": reason,
        "thresholds": {
            "source_margin": source_margin,
            "target_margin": target_margin,
        },
    }


def blip_attack_decision(
    original_caption: str,
    adversarial_caption: str,
    source_text: str,
    target_text: str | None,
    targeted: bool = True,
    target_overlap_threshold: float = 0.20,
    source_drop_threshold: float = 0.05,
) -> Dict[str, float | bool | str]:
    """Deterministic BLIP attack decision based on lexical overlap."""
    source_text = source_text or ""
    target_text = target_text or ""
    orig_tokens = tokenize_content(original_caption)
    adv_tokens = tokenize_content(adversarial_caption)
    source_tokens = tokenize_content(source_text)
    target_tokens = tokenize_content(target_text)

    orig_source_overlap = jaccard_similarity(orig_tokens, source_tokens)
    adv_source_overlap = jaccard_similarity(adv_tokens, source_tokens)
    orig_target_overlap = jaccard_similarity(orig_tokens, target_tokens)
    adv_target_overlap = jaccard_similarity(adv_tokens, target_tokens)
    source_overlap_drop = orig_source_overlap - adv_source_overlap
    target_overlap_gain = adv_target_overlap - orig_target_overlap
    caption_changed = normalize_text(original_caption) != normalize_text(adversarial_caption)
    target_keyword_hit = bool(set(target_tokens) & set(adv_tokens)) if target_tokens else False

    if targeted:
        success = (
            caption_changed
            and target_keyword_hit
            and adv_target_overlap >= max(target_overlap_threshold, orig_target_overlap)
            and adv_target_overlap >= adv_source_overlap
        )
        reason = _join_reason([
            f"目标关键词命中={target_keyword_hit}",
            f"目标重叠提升 {target_overlap_gain:.4f}",
            f"源重叠下降 {source_overlap_drop:.4f}",
        ])
    else:
        success = caption_changed and source_overlap_drop >= source_drop_threshold
        reason = _join_reason([
            f"描述变化={caption_changed}",
            f"源重叠下降 {source_overlap_drop:.4f}",
        ])

    return {
        "task": "blip",
        "targeted": targeted,
        "attack_success": bool(success),
        "caption_changed": caption_changed,
        "target_keyword_hit": target_keyword_hit,
        "orig_source_overlap": orig_source_overlap,
        "adv_source_overlap": adv_source_overlap,
        "orig_target_overlap": orig_target_overlap,
        "adv_target_overlap": adv_target_overlap,
        "source_overlap_drop": source_overlap_drop,
        "target_overlap_gain": target_overlap_gain,
        "rule": "targeted: caption_changed and target_keyword_hit and adv_target_overlap>=max(threshold, orig_target_overlap) and adv_target_overlap>=adv_source_overlap; untargeted: caption_changed and source_overlap_drop>=threshold",
        "reason": reason,
        "thresholds": {
            "target_overlap_threshold": target_overlap_threshold,
            "source_drop_threshold": source_drop_threshold,
        },
    }
