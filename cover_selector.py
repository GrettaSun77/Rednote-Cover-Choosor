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

DIMENSIONS = [
    "eye_catch",
    "cover_fit",
    "subject_clarity",
    "mood",
    "composition",
]

DIMENSION_KEYWORDS = {
    "eye_catch": ["吸引", "点进来", "抓眼", "一眼", "绝", "美爆", "侵略性", "张力", "mvp"],
    "cover_fit": ["封面", "头像", "首图", "适合", "点击", "高级", "杂志", "专辑"],
    "subject_clarity": ["清楚", "清晰", "干净", "脸", "五官", "突出", "清纯"],
    "mood": ["氛围", "故事", "自然", "松弛", "情绪", "忧郁", "时尚", "人生照片"],
    "composition": ["构图", "完整", "平衡", "比例", "舒服", "主体"],
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
                weights[dimension] += sum(reason.count(keyword) for keyword in keywords)

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


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_0_10(value: float) -> float:
    return round(clamp(value, 0.0, 10.0), 2)


def compute_colorfulness(image: Image.Image) -> float:
    r, g, b = image.split()
    rg = ImageChops.difference(r, g)
    yb = ImageChops.difference(ImageChops.add_modulo(r, g), b)
    return math.sqrt(ImageStat.Stat(rg).mean[0] ** 2 + ImageStat.Stat(yb).mean[0] ** 2)


def score_image_features(uploaded_images: list[UploadedImage]) -> list[ScoreCard]:
    cards: list[ScoreCard] = []
    for image in uploaded_images:
        pil_image = pil_image_from_bytes(image.data)
        grayscale = pil_image.convert("L")
        stat = ImageStat.Stat(grayscale)
        brightness = stat.mean[0] / 255.0
        contrast = stat.stddev[0] / 64.0
        edges = grayscale.filter(ImageFilter.FIND_EDGES)
        sharpness = ImageStat.Stat(edges).mean[0] / 32.0
        width, height = pil_image.size
        ratio = width / max(height, 1)
        colorfulness = compute_colorfulness(pil_image) / 20.0

        details = {
            "brightness": normalize_0_10(brightness * 10),
            "contrast": normalize_0_10(contrast * 10),
            "sharpness": normalize_0_10(sharpness * 10),
            "colorfulness": normalize_0_10(colorfulness),
            "thumbnail_balance": normalize_0_10((1 - abs(ratio - 0.8)) * 10),
        }
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


def build_openai_messages(historical_profile: dict[str, Any], uploaded_images: list[UploadedImage]) -> list[dict[str, Any]]:
    examples = {
        "dimension_weights": historical_profile["dimension_weights"],
        "winner_reason_samples": historical_profile["winner_reason_samples"],
        "task": "Rank candidate images for Xiaohongshu cover selection.",
    }
    user_content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "Review these candidate cover images for Xiaohongshu. Use the historical preference profile "
                "to score each image on eye_catch, cover_fit, subject_clarity, mood, and composition. "
                "Return strict JSON only."
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
                        "You are a vision ranking model for Xiaohongshu cover selection. "
                        "Prioritize likely public preference and click-through appeal, not pure artistic merit."
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
                                    "reason": {"type": "string"},
                                },
                                "required": [
                                    "image_index",
                                    "eye_catch",
                                    "cover_fit",
                                    "subject_clarity",
                                    "mood",
                                    "composition",
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
        }
        cards.append(
            ScoreCard(
                image_index=int(item["image_index"]),
                source="openai_vision",
                total_score=round(statistics.mean(details.values()), 2),
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
        cards.append(
            ScoreCard(
                image_index=card.image_index,
                source="history_alignment",
                total_score=round(sum(weighted.values()) * 10, 2),
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
) -> SelectionResult:
    openai_map = index_cards(openai_cards)
    feature_map = index_cards(feature_cards)
    history_map = index_cards(history_cards)

    final_scores: list[dict[str, Any]] = []
    for image in uploaded_images:
        openai_score = openai_map.get(image.index).total_score if image.index in openai_map else 0.0
        feature_score = feature_map.get(image.index).total_score if image.index in feature_map else 0.0
        history_score = history_map.get(image.index).total_score if image.index in history_map else 0.0
        final_score = round(0.55 * openai_score + 0.2 * feature_score + 0.25 * history_score, 2)
        final_scores.append(
            {
                "image_index": image.index,
                "final_score": final_score,
                "openai_score": round(openai_score, 2),
                "feature_score": round(feature_score, 2),
                "history_score": round(history_score, 2),
                "openai_reason": openai_map.get(image.index).summary if image.index in openai_map else "暂无",
                "feature_reason": feature_map.get(image.index).summary if image.index in feature_map else "暂无",
            }
        )

    final_scores.sort(key=lambda item: item["final_score"], reverse=True)
    best = final_scores[0]
    second_score = final_scores[1]["final_score"] if len(final_scores) > 1 else max(best["final_score"] - 1, 0)
    confidence = round(clamp(0.55 + (best["final_score"] - second_score) / 10, 0.0, 0.99), 2)

    return SelectionResult(
        best_image_index=best["image_index"],
        confidence=confidence,
        reason=(
            f"第 {best['image_index']} 张在 OpenAI 视觉评分、本地图片特征评分和历史偏好校准的融合结果中排名最高。"
        ),
        final_scores=final_scores,
        source_scores={
            "openai_vision": [card.__dict__ for card in openai_cards],
            "local_image_features": [card.__dict__ for card in feature_cards],
            "history_alignment": [card.__dict__ for card in history_cards],
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
    result = blend_scores(uploaded_images, openai_cards, feature_cards, history_cards)
    result.historical_profile = historical_profile
    return result
