"""Streamlit demo: score a face image with the trained OFIQ quality CNN."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from web.inference import QualityScorer

GITHUB = "https://github.com/inezaodon/image-quality-cnn"
ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Face Image Quality Scorer",
    page_icon="👤",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading quality model…")
def load_scorer() -> QualityScorer:
    return QualityScorer(ROOT / "best_model_full.pt")


def main() -> None:
    st.title("Face Image Quality Scorer")
    st.caption("SmallResNet CNN · OFIQ UnifiedQualityScore.native")

    st.markdown(
        """
Upload a face photo to get an instant **biometric quality** score — the same kind of
metric [OFIQ](https://www.bsi.bund.de/OFIQ) computes, approximated by a compact network
trained on 70,000 FFHQ images.

**Not a beauty rating.** Higher = better usability for face recognition
(sharpness, pose, lighting, occlusion). Most faces land around **20–26** on this
~11–33 native scale; scores above ~28 are uncommon.
"""
    )

    scorer = load_scorer()

    uploaded = st.file_uploader(
        "Face image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        help="Phone or web photos work — we detect and crop the face first.",
    )

    if uploaded is None:
        st.info("Drop an image above to score it.")
    else:
        raw = uploaded.getvalue()
        original = Image.open(uploaded).convert("RGB")

        with st.spinner("Detecting face and scoring…"):
            try:
                result = scorer.score_image_bytes(raw)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not score image: {exc}")
                return

        left, right = st.columns([1, 1.2])
        with left:
            st.subheader("Your upload")
            st.image(original, use_container_width=True)
            import base64
            import io

            crop = Image.open(io.BytesIO(base64.b64decode(result.crop_preview_b64)))
            st.subheader("What the model scored")
            st.image(crop, width=220)
            st.caption(result.crop_note)

        with right:
            st.metric("Quality score", f"{result.score:.2f}", help="Native OFIQ units")
            st.markdown(f"**{result.band}** — {result.band_summary}")
            st.progress(min(1.0, max(0.0, result.percentile / 100.0)))
            st.caption(
                f"{result.percentile:.0f}th percentile of training range "
                f"({result.lo:.1f}–{result.hi:.1f}). Typical FFHQ faces sit near the middle."
            )
            st.markdown(result.band_detail)

            with st.expander("How to read this score"):
                st.markdown(
                    f"""
- **Scale ~{result.lo:.0f}–{result.hi:.0f}**, not 0–100.
- **~20–26** = typical / fine for many uses.
- **Above ~28** = uncommon; **above 30** ≈ top 1% of training labels.
- Model error on held-out FFHQ: about **±{result.mae:.2f}** (Pearson r ≈ 0.87).
- Checkpoint: `best_model_full.pt` (epoch {result.epoch}).
"""
                )

    st.divider()
    st.subheader("About this project")
    st.markdown(
        """
OFIQ measures face-image quality accurately but slowly. This project trains a
**SmallResNet** (~0.33M parameters) to predict `UnifiedQualityScore.native` in
milliseconds for large-scale use.
"""
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.link_button("Project report (REPORT.md)", f"{GITHUB}/blob/main/REPORT.md")
    with c2:
        st.link_button("Technical report (PDF)", f"{GITHUB}/blob/main/Technical%20Reports/tech_report.pdf")
    with c3:
        st.link_button("Source on GitHub", GITHUB)


if __name__ == "__main__":
    main()
