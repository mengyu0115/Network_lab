"""
Output-level metrics for black-box multimodal model evaluation.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def answer_similarity(original_answer: str, adversarial_answer: str) -> float:
    return float(
        SequenceMatcher(
            None,
            normalize_answer(original_answer),
            normalize_answer(adversarial_answer),
        ).ratio()
    )


def answer_changed(original_answer: str, adversarial_answer: str, threshold: float = 0.85) -> bool:
    return answer_similarity(original_answer, adversarial_answer) < threshold


def keyword_hit_rate(answer: str, keywords: Iterable[str]) -> float:
    keywords = [kw.strip().lower() for kw in keywords if kw.strip()]
    if not keywords:
        return 0.0
    normalized = normalize_answer(answer)
    hits = sum(1 for kw in keywords if kw in normalized)
    return float(hits / len(keywords))


def contains_any_keyword(answer: str, keywords: Iterable[str]) -> bool:
    keywords = [kw.strip().lower() for kw in keywords if kw.strip()]
    if not keywords:
        return False
    normalized = normalize_answer(answer)
    return any(kw in normalized for kw in keywords)


def extract_key_value_fields(answer: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*\d\.\、\)\s]+", "", line)
        pairs = _extract_inline_pairs(line)
        if pairs:
            fields.update(pairs)
            continue
        if "：" in line:
            key, value = line.split("：", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        elif "=" in line:
            key, value = line.split("=", 1)
        else:
            continue
        key = normalize_answer(key).strip(" ：:=")
        value = normalize_answer(value).strip(" ：:=")
        if key and value:
            fields[key] = value
    return fields


def _extract_inline_pairs(text: str) -> Dict[str, str]:
    pairs: Dict[str, str] = {}
    pattern = re.compile(r"([\u4e00-\u9fa5A-Za-z0-9_\-]+)\s*[=:：]\s*([^，,；;\s]+)")
    for key, value in pattern.findall(text):
        key_norm = normalize_answer(key).strip(" ：:=")
        value_norm = normalize_answer(value).strip(" ：:=")
        if key_norm and value_norm:
            pairs[key_norm] = value_norm

    # Common OCR table output: 张三=数学=89, 李四-英语-76, 王五 计算机 92.
    row_pattern = re.compile(
        r"([\u4e00-\u9fa5]{2,4})\s*[=\-－\s]\s*([\u4e00-\u9fa5A-Za-z]+)\s*[=\-－\s]\s*(\d+(?:\.\d+)?)"
    )
    for name, course, score in row_pattern.findall(text):
        pairs[normalize_answer(name)] = score
        pairs[f"{normalize_answer(name)}_课程"] = normalize_answer(course)
        pairs[f"{normalize_answer(name)}_成绩"] = score
    return pairs


def parse_ground_truth_fields(text: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for raw_line in text.replace("；", "\n").replace(";", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif "：" in line:
            key, value = line.split("：", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        key = normalize_answer(key).strip(" ：:=")
        value = normalize_answer(value).strip(" ：:=")
        if key and value:
            fields[key] = value
    return fields


def extract_numbers(answer: str) -> List[str]:
    return re.findall(r"\d+(?:\.\d+)?", answer)


def _value_matches(expected: str, actual_text: str) -> bool:
    expected_norm = normalize_answer(expected)
    actual_norm = normalize_answer(actual_text)
    expected_numbers = extract_numbers(expected_norm)
    if expected_numbers:
        actual_numbers = extract_numbers(actual_norm)
        return all(number in actual_numbers for number in expected_numbers)
    return expected_norm in actual_norm


def _keys_match(expected_key: str, answer_key: str) -> bool:
    expected_key = normalize_answer(expected_key)
    answer_key = normalize_answer(answer_key)
    if expected_key == answer_key:
        return True
    if expected_key in answer_key or answer_key in expected_key:
        return True
    if "_" in answer_key:
        base, suffix = answer_key.rsplit("_", 1)
        if expected_key == base or expected_key == suffix:
            return True
    return False


def evaluate_against_ground_truth(answer: str, ground_truth_fields: Dict[str, str]) -> Dict:
    answer_fields = extract_key_value_fields(answer)
    correct_fields: List[str] = []
    wrong_fields: List[str] = []

    for expected_key, expected_value in ground_truth_fields.items():
        matched_values = []
        for answer_key, answer_value in answer_fields.items():
            if _keys_match(expected_key, answer_key):
                matched_values.append(answer_value)
        candidate_text = "\n".join(matched_values) if matched_values else answer
        if _value_matches(expected_value, candidate_text):
            correct_fields.append(expected_key)
        else:
            wrong_fields.append(expected_key)

    total = len(ground_truth_fields)
    accuracy = float(len(correct_fields) / max(total, 1) * 100.0)
    return {
        "field_accuracy": accuracy,
        "correct_fields": correct_fields,
        "wrong_fields": wrong_fields,
    }


def structured_output_diff(original_answer: str, adversarial_answer: str) -> Dict:
    original_fields = extract_key_value_fields(original_answer)
    adversarial_fields = extract_key_value_fields(adversarial_answer)
    shared_keys = sorted(set(original_fields) & set(adversarial_fields))
    changed_fields = [
        key
        for key in shared_keys
        if original_fields[key] != adversarial_fields[key]
    ]

    original_numbers = extract_numbers(original_answer)
    adversarial_numbers = extract_numbers(adversarial_answer)
    missing_numbers = [
        number
        for number in original_numbers
        if number not in adversarial_numbers
    ]
    new_numbers = [
        number
        for number in adversarial_numbers
        if number not in original_numbers
    ]

    return {
        "structured_field_count": len(shared_keys),
        "structured_field_changed_count": len(changed_fields),
        "structured_field_change_rate": float(len(changed_fields) / max(len(shared_keys), 1) * 100.0),
        "changed_fields": changed_fields,
        "missing_numbers": missing_numbers,
        "new_numbers": new_numbers,
        "number_change_count": len(missing_numbers) + len(new_numbers),
    }


def evaluate_blackbox_pair(
    original_answer: str,
    adversarial_answer: str,
    expected_keywords: List[str] | None = None,
    target_keywords: List[str] | None = None,
    ground_truth_fields: Dict[str, str] | None = None,
    change_threshold: float = 0.85,
    strict_original_accuracy_threshold: float = 80.0,
) -> Dict:
    similarity = answer_similarity(original_answer, adversarial_answer)
    changed = similarity < change_threshold
    expected_drop = 0.0
    target_gain = 0.0
    if expected_keywords:
        expected_drop = keyword_hit_rate(original_answer, expected_keywords) - keyword_hit_rate(
            adversarial_answer,
            expected_keywords,
        )
    if target_keywords:
        target_gain = keyword_hit_rate(adversarial_answer, target_keywords) - keyword_hit_rate(
            original_answer,
            target_keywords,
        )

    expected_missing = bool(expected_keywords) and not contains_any_keyword(adversarial_answer, expected_keywords)
    target_hit = bool(target_keywords) and contains_any_keyword(adversarial_answer, target_keywords)
    structured_diff = structured_output_diff(original_answer, adversarial_answer)
    structured_changed = (
        structured_diff["structured_field_changed_count"] > 0
        or structured_diff["number_change_count"] > 0
    )
    ground_truth_metrics = {}
    ground_truth_success = False
    if ground_truth_fields:
        original_truth = evaluate_against_ground_truth(original_answer, ground_truth_fields)
        adversarial_truth = evaluate_against_ground_truth(adversarial_answer, ground_truth_fields)
        degraded_fields = [
            field
            for field in original_truth["correct_fields"]
            if field in adversarial_truth["wrong_fields"]
        ]
        original_accuracy = original_truth["field_accuracy"]
        adversarial_accuracy = adversarial_truth["field_accuracy"]
        accuracy_drop = original_accuracy - adversarial_accuracy
        strict_original_ready = original_accuracy >= strict_original_accuracy_threshold
        ground_truth_success = bool(degraded_fields) and strict_original_ready and accuracy_drop > 0
        ground_truth_metrics = {
            "ground_truth_field_count": len(ground_truth_fields),
            "original_field_accuracy": original_accuracy,
            "adversarial_field_accuracy": adversarial_accuracy,
            "ground_truth_accuracy_drop": accuracy_drop,
            "strict_original_accuracy_threshold": strict_original_accuracy_threshold,
            "strict_original_ready": strict_original_ready,
            "original_correct_fields": original_truth["correct_fields"],
            "adversarial_correct_fields": adversarial_truth["correct_fields"],
            "degraded_ground_truth_fields": degraded_fields,
            "ground_truth_attack_success": ground_truth_success,
        }

    migration_success = (
        ground_truth_success
        or (not ground_truth_fields and (changed or expected_drop > 0 or target_gain > 0 or expected_missing or target_hit or structured_changed))
    )
    return {
        "answer_similarity": similarity,
        "answer_change_rate": 100.0 if changed else 0.0,
        "answer_changed": changed,
        "expected_keyword_drop": expected_drop,
        "expected_keyword_missing": expected_missing,
        "target_keyword_gain": target_gain,
        "target_keyword_hit": target_hit,
        **structured_diff,
        "structured_output_changed": structured_changed,
        **ground_truth_metrics,
        "blackbox_transfer_success": bool(migration_success),
        "blackbox_transfer_success_rate": 100.0 if migration_success else 0.0,
    }
