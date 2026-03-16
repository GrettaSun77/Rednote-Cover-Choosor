from __future__ import annotations

import base64
import json
import math
import os
import statistics
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageFilter, ImageStat


DATASET_PATH = Path(__file__).resolve().parent / "data" / "processed" / "training_dataset.json"
EMBEDDED_MEDIA_DIR = Path(__file__).resolve().parent / "data" / "processed" / "embedded_media"

DIMENSIONS = [
    "eye_catch",
    "cover_fit",
    "subject_clarity",
    "mood",
    "composition",
    "xiaohongshu_fit",
    "rigid_penalty",
]

DIMENSION_KEYWORDS = {
    "eye_catch": ["吸引", "点进来", "抓眼", "一眼", "绝", "美爆", "侵略性", "张力", "mvp"],
    "cover_fit": ["封面", "头像", "首图", "适合", "点击", "高级", "杂志", "专辑"],
    "subject_clarity": ["清楚", "清晰", "干净", "脸", "五官", "突出", "清纯"],
    "mood": ["氛围", "故事", "自然", "松弛", "情绪", "忧郁", "时尚", "人生照片"],
    "composition": ["构图", "完整", "平衡", "比例", "舒服", "主体"],
    "xiaohongshu_fit": ["封面", "点进来", "吸引", "氛围", "自然", "高级", "杂志", "专辑"],
    "rigid_penalty": ["证件照", "职业照", "简历照", "工牌", "太正式", "死板"],
}


@dataclass
class UploadedImage:
    index: int
    name: str
    mime_type: str
    data: bytes


@dataclass
class ScoreCard:
    image_index: int
    source: str
    total_score: float
    summary: str
    details: dict[str, float]


@dataclass
class SelectionResult:
    best_image_index: int
    confidence: float
    reason: str
    final_scores: list[dict[str, Any]]
    source_scores: dict[str, list[dict[str, Any]]]
    historical_profile: dict[str, Any]


def load_dataset() -> dict[str, Any]:
    if not DATASET_PATH.exists():
        return {"records": []}
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    try:
        import streamlit as st

        secret_value = st.secrets.get("OPENAI_API_KEY")
        if secret_value:
            return str(secret_value)
    except Exception:
        pass

    return ""


def load_historical_profile(dataset: dict[str, Any]) -> dict[str, Any]:
    weights = {dimension: 1.0 for dimension in DIMENSIONS}
    winner_reason_texts: list[str] = []

    for record in dataset.get("records", []):
        for candidate in record.get("candidates", []):
            if not candidate.get("winner"):
                continue
            reason = candidate.get("reason_summary", "")
            winner_reason_texts.append(reason)
            for dimension, keywords in DIMENSION_KEYWORDS.items():
                hit_count = sum(reason.count(keyword) for keyword in keywords)
                if dimension == "rigid_penalty":
                    weights[dimension] += max(0, 1 - hit_count)
                else:
                    weights[dimension] += hit_count

    total = sum(weights.values()) or 1.0
    return {
        "dimension_weights": {key: round(value / total, 4) for key, value in weights.items()},
        "winner_reason_samples": winner_reason_texts[:10],
        "record_count": len(dataset.get("records", [])),
    }


def image_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def pil_image_from_bytes(image_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(image_bytes))
    return image.convert("RGB")


def pil_image_from_path(image_path: Path) -> Image.Image:
    image = Image.open(image_path)
    return image.convert("RGB")


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_0_10(value: float) -> float:
    return round(clamp(value, 0.0, 10.0), 2)


def compute_colorfulness(image: Image.Image) -> float:
    r, g, b = image.split()
    rg = ImageChops.difference(r, g)
    yb = ImageChops.difference(ImageChops.add_modulo(r, g), b)
    return math.sqrt(ImageStat.Stat(rg).mean[0] ** 2 + ImageStat.Stat(yb).mean[0] ** 2)


def extract_visual_features(image: Image.Image) -> dict[str, float]:
    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    brightness = stat.mean[0] / 255.0
    contrast = stat.stddev[0] / 64.0
    edges = grayscale.filter(ImageFilter.FIND_EDGES)
    sharpness = ImageStat.Stat(edges).mean[0] / 32.0
    width, height = image.size
    ratio = width / max(height, 1)
    colorfulness = compute_colorfulness(image) / 20.0
    return {
        "brightness": normalize_0_10(brightness * 10),
        "contrast": normalize_0_10(contrast * 10),
        "sharpness": normalize_0_10(sharpness * 10),
        "colorfulness": normalize_0_10(colorfulness),
        "thumbnail_balance": normalize_0_10((1 - abs(ratio - 0.8)) * 10),
    }


def average_feature_maps(feature_maps: list[dict[str, float]]) -> dict[str, float]:
    if not feature_maps:
        return {}
    keys = feature_maps[0].keys()
    return {
        key: round(sum(feature_map[key] for feature_map in feature_maps) / len(feature_maps), 4)
        for key in keys
    }


def feature_distance(a: dict[str, float], b: dict[str, float]) -> float:
    common_keys = [key for key in a.keys() if key in b]
    if not common_keys:
        return 999.0
    return math.sqrt(sum((a[key] - b[key]) ** 2 for key in common_keys) / len(common_keys))


def map_historical_images(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    media_files = sorted(EMBEDDED_MEDIA_DIR.glob("image*.jpeg"), key=lambda path: int(re.search(r"(\d+)", path.stem).group(1)))
    mapped: list[dict[str, Any]] = []
    media_index = 0
    for record in dataset.get("records", []):
        for candidate in record.get("candidates", []):
            if media_index >= len(media_files):
                return mapped
            mapped.append(
                {
                    "batch_name": record.get("batch_name"),
                    "image_label": candidate.get("image_label"),
                    "winner": bool(candidate.get("winner")),
                    "path": media_files[media_index],
                }
            )
            media_index += 1
    return mapped


def load_historical_visual_profile(dataset: dict[str, Any]) -> dict[str, Any]:
    mapped_images = map_historical_images(dataset)
    if not mapped_images:
        return {"winner_centroid": {}, "non_winner_centroid": {}, "winner_count": 0, "non_winner_count": 0}

    winner_features: list[dict[str, float]] = []
    non_winner_features: list[dict[str, float]] = []
    for item in mapped_images:
        features = extract_visual_features(pil_image_from_path(item["path"]))
        if item["winner"]:
            winner_features.append(features)
        else:
            non_winner_features.append(features)

    return {
        "winner_centroid": average_feature_maps(winner_features),
        "non_winner_centroid": average_feature_maps(non_winner_features),
        "winner_count": len(winner_features),
        "non_winner_count": len(non_winner_features),
    }


def score_image_features(uploaded_images: list[UploadedImage]) -> list[ScoreCard]:
    cards: list[ScoreCard] = []
    for image in uploaded_images:
        pil_image = pil_image_from_bytes(image.data)
        details = extract_visual_features(pil_image)
        total_score = round(
            0.18 * details["brightness"]
            + 0.22 * details["contrast"]
            + 0.24 * details["sharpness"]
            + 0.18 * details["colorfulness"]
            + 0.18 * details["thumbnail_balance"],
            2,
        )
        cards.append(
            ScoreCard(
                image_index=image.index,
                source="local_image_features",
                total_score=total_score,
                summary=(
                    f"brightness {details['brightness']}, contrast {details['contrast']}, "
                    f"sharpness {details['sharpness']}, colorfulness {details['colorfulness']}"
                ),
                details=details,
            )
        )
    return cards


def score_historical_visual_match(dataset: dict[str, Any], uploaded_images: list[UploadedImage]) -> list[ScoreCard]:
    visual_profile = load_historical_visual_profile(dataset)
    winner_centroid = visual_profile.get("winner_centroid", {})
    non_winner_centroid = visual_profile.get("non_winner_centroid", {})
    cards: list[ScoreCard] = []
    for image in uploaded_images:
        details = extract_visual_features(pil_image_from_bytes(image.data))
        winner_distance = feature_distance(details, winner_centroid) if winner_centroid else 999.0
        non_winner_distance = feature_distance(details, non_winner_centroid) if non_winner_centroid else 999.0
        raw_score = 5 + (non_winner_distance - winner_distance) * 2.2
        total_score = normalize_0_10(raw_score)
        summary = (
            f"与历史高票图距离 {winner_distance:.2f}，与历史普通图距离 {non_winner_distance:.2f}。"
            "分数越高表示越接近历史 winner 的网感。"
        )
        cards.append(
            ScoreCard(
                image_index=image.index,
                source="historical_visual_match",
                total_score=total_score,
                summary=summary,
                details={
                    "winner_distance": round(winner_distance, 4),
                    "non_winner_distance": round(non_winner_distance, 4),
                },
            )
        )
    return cards


def build_openai_messages(historical_profile: dict[str, Any], uploaded_images: list[UploadedImage]) -> list[dict[str, Any]]:
    examples = {
        "dimension_weights": historical_profile["dimension_weights"],
        "winner_reason_samples": historical_profile["winner_reason_samples"],
        "task": "为小红书头图场景对候选图片进行排序。",
    }
    user_content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "请为这些小红书候选头图打分。你需要结合历史偏好画像，"
                "从吸睛度、封面感、主体清晰度、情绪氛围、构图完整度、小红书平台适配度六个维度评分。"
                "如果图片过于像证件照、简历照、职业形象照、工牌照或商务宣传照，要明确扣分。"
                "请优先选择更像社交平台头图、能让人想点进来的图，而不是最标准正式的照片。"
                "所有说明必须使用简体中文。只返回严格 JSON。"
            ),
        },
        {"type": "input_text", "text": json.dumps(examples, ensure_ascii=False)},
    ]

    for image in uploaded_images:
        user_content.append({"type": "input_text", "text": f"candidate_image_{image.index}"})
        user_content.append({"type": "input_image", "image_url": image_to_data_url(image.data, image.mime_type)})

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "你是一个用于小红书头图选择的视觉排序助手。"
                        "你的目标不是判断哪张图艺术性最高，而是判断哪张图最可能获得大众偏好和点击。"
                        "你必须避免把过于正式、过于像证件照或职业照的图片排在前面，除非它同时具备明显的头图吸引力。"
                        "所有输出说明都必须使用简体中文。"
                    ),
                }
            ],
        },
        {"role": "user", "content": user_content},
    ]


def score_with_openai(historical_profile: dict[str, Any], uploaded_images: list[UploadedImage]) -> list[ScoreCard]:
    api_key = get_api_key()
    if not api_key:
        return []

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini"),
        input=build_openai_messages(historical_profile, uploaded_images),
        text={
            "format": {
                "type": "json_schema",
                "name": "xhs_cover_ranking",
                "schema": {
                    "type": "object",
                    "properties": {
                        "images": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "image_index": {"type": "integer"},
                                    "eye_catch": {"type": "number"},
                                    "cover_fit": {"type": "number"},
                                    "subject_clarity": {"type": "number"},
                                    "mood": {"type": "number"},
                                    "composition": {"type": "number"},
                                    "xiaohongshu_fit": {"type": "number"},
                                    "rigid_penalty": {"type": "number"},
                                    "reason": {"type": "string"},
                                },
                                "required": [
                                    "image_index",
                                    "eye_catch",
                                    "cover_fit",
                                    "subject_clarity",
                                    "mood",
                                    "composition",
                                    "xiaohongshu_fit",
                                    "rigid_penalty",
                                    "reason",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["images"],
                    "additionalProperties": False,
                },
            }
        },
    )

    payload = json.loads(response.output_text)
    cards: list[ScoreCard] = []
    for item in payload["images"]:
        details = {
            "eye_catch": normalize_0_10(item["eye_catch"]),
            "cover_fit": normalize_0_10(item["cover_fit"]),
            "subject_clarity": normalize_0_10(item["subject_clarity"]),
            "mood": normalize_0_10(item["mood"]),
            "composition": normalize_0_10(item["composition"]),
            "xiaohongshu_fit": normalize_0_10(item["xiaohongshu_fit"]),
            "rigid_penalty": normalize_0_10(item["rigid_penalty"]),
        }
        total_score = round(
            (
                details["eye_catch"] * 0.2
                + details["cover_fit"] * 0.16
                + details["subject_clarity"] * 0.12
                + details["mood"] * 0.14
                + details["composition"] * 0.1
                + details["xiaohongshu_fit"] * 0.28
                - details["rigid_penalty"] * 0.18
            ),
            2,
        )
        cards.append(
            ScoreCard(
                image_index=int(item["image_index"]),
                source="openai_vision",
                total_score=total_score,
                summary=item["reason"],
                details=details,
            )
        )
    return cards


def score_history_alignment(historical_profile: dict[str, Any], openai_cards: list[ScoreCard]) -> list[ScoreCard]:
    weights = historical_profile.get("dimension_weights", {})
    cards: list[ScoreCard] = []
    for card in openai_cards:
        weighted = {
            dimension: round(card.details.get(dimension, 0.0) * weights.get(dimension, 0.0), 4)
            for dimension in DIMENSIONS
        }
        weighted_total = sum(weighted.get(dimension, 0.0) for dimension in DIMENSIONS if dimension != "rigid_penalty")
        weighted_total -= weighted.get("rigid_penalty", 0.0)
        cards.append(
            ScoreCard(
                image_index=card.image_index,
                source="history_alignment",
                total_score=round(weighted_total * 10, 2),
                summary="Weighted by historical winner reasons.",
                details=weighted,
            )
        )
    return cards


def index_cards(cards: list[ScoreCard]) -> dict[int, ScoreCard]:
    return {card.image_index: card for card in cards}


def blend_scores(
    uploaded_images: list[UploadedImage],
    openai_cards: list[ScoreCard],
    feature_cards: list[ScoreCard],
    history_cards: list[ScoreCard],
    visual_match_cards: list[ScoreCard],
) -> SelectionResult:
    openai_map = index_cards(openai_cards)
    feature_map = index_cards(feature_cards)
    history_map = index_cards(history_cards)
    visual_match_map = index_cards(visual_match_cards)

    final_scores: list[dict[str, Any]] = []
    for image in uploaded_images:
        openai_score = openai_map.get(image.index).total_score if image.index in openai_map else 0.0
        feature_score = feature_map.get(image.index).total_score if image.index in feature_map else 0.0
        history_score = history_map.get(image.index).total_score if image.index in history_map else 0.0
        visual_match_score = visual_match_map.get(image.index).total_score if image.index in visual_match_map else 0.0
        final_score = round(0.38 * openai_score + 0.14 * feature_score + 0.18 * history_score + 0.3 * visual_match_score, 2)
        final_scores.append(
            {
                "候选图": image.index,
                "最终分": final_score,
                "视觉模型分": round(openai_score, 2),
                "图片特征分": round(feature_score, 2),
                "历史偏好分": round(history_score, 2),
                "历史网感分": round(visual_match_score, 2),
                "视觉分析说明": openai_map.get(image.index).summary if image.index in openai_map else "暂无",
                "图片特征说明": feature_map.get(image.index).summary if image.index in feature_map else "暂无",
                "历史网感说明": visual_match_map.get(image.index).summary if image.index in visual_match_map else "暂无",
            }
        )

    final_scores.sort(key=lambda item: item["最终分"], reverse=True)
    best = final_scores[0]
    second_score = final_scores[1]["最终分"] if len(final_scores) > 1 else max(best["最终分"] - 1, 0)
    confidence = round(clamp(0.55 + (best["最终分"] - second_score) / 10, 0.0, 0.99), 2)

    return SelectionResult(
        best_image_index=best["候选图"],
        confidence=confidence,
        reason=(
            f"第 {best['候选图']} 张在视觉模型评分、图片特征评分和历史偏好校准的融合结果中排名最高。"
            f"同时它与历史高票图片的网感风格更接近。"
        ),
        final_scores=final_scores,
        source_scores={
            "openai_vision": [card.__dict__ for card in openai_cards],
            "local_image_features": [card.__dict__ for card in feature_cards],
            "history_alignment": [card.__dict__ for card in history_cards],
            "historical_visual_match": [card.__dict__ for card in visual_match_cards],
        },
        historical_profile={},
    )


def run_cover_selection(dataset: dict[str, Any], uploaded_images: list[UploadedImage]) -> SelectionResult:
    historical_profile = load_historical_profile(dataset)
    openai_cards = score_with_openai(historical_profile, uploaded_images)
    if not openai_cards:
        raise RuntimeError("OPENAI_API_KEY is missing, so the visual scoring layer cannot run.")
    feature_cards = score_image_features(uploaded_images)
    history_cards = score_history_alignment(historical_profile, openai_cards)
    visual_match_cards = score_historical_visual_match(dataset, uploaded_images)
    result = blend_scores(uploaded_images, openai_cards, feature_cards, history_cards, visual_match_cards)
    result.historical_profile = historical_profile
    result.historical_profile["historical_visual_profile"] = load_historical_visual_profile(dataset)
    return result
