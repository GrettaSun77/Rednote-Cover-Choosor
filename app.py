from __future__ import annotations

from typing import Any

import streamlit as st

from cover_selector import UploadedImage, load_dataset, run_cover_selection


def to_uploaded_images(files: list[Any]) -> list[UploadedImage]:
    images: list[UploadedImage] = []
    for index, file in enumerate(files, start=1):
        images.append(
            UploadedImage(
                index=index,
                name=file.name,
                mime_type=file.type or "image/jpeg",
                data=file.getvalue(),
            )
        )
    return images


st.set_page_config(page_title="小红书头图选择器", page_icon="CC", layout="wide")
st.title("小红书头图选择器 MVP")

dataset = load_dataset()
record_count = len(dataset.get("records", []))
st.caption(f"已加载历史样本批次：{record_count}")

with st.sidebar:
    st.subheader("评分流程")
    st.write("1. OpenAI 视觉评分")
    st.write("2. 本地图片特征评分")
    st.write("3. 历史偏好重排序")
    st.write("4. 加权融合")
    st.divider()
    st.write(f"历史样本批次：{record_count}")
    st.write("建议每次上传 3 到 6 张候选图")

files = st.file_uploader(
    "上传候选图片",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if files:
    cols = st.columns(min(len(files), 4))
    for index, file in enumerate(files):
        with cols[index % len(cols)]:
            st.image(file, caption=f"候选图 {index + 1}", width="stretch")

    if st.button("开始选择头图", type="primary"):
        try:
            result = run_cover_selection(dataset, to_uploaded_images(files))
        except Exception as exc:
            st.error(str(exc))
        else:
            st.success(f"推荐头图：第 {result.best_image_index} 张")
            st.write(f"置信度：{result.confidence}")
            st.write(result.reason)

            st.subheader("最终融合排序")
            st.dataframe(result.final_scores, use_container_width=True)

            st.subheader("历史偏好画像")
            st.json(result.historical_profile, expanded=False)

            st.subheader("各评分源明细")
            st.json(result.source_scores, expanded=False)
else:
    st.info("先上传一组候选图片，再运行融合评分流程。")
