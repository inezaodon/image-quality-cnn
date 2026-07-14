"""Streamlit demo matching the Flask web UI design."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import streamlit as st
from PIL import Image

from web.inference import QualityScorer

GITHUB = "https://github.com/inezaodon/image-quality-cnn"
ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Face Image Quality Scorer",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=Instrument+Serif:ital@0;1&display=swap');

html, body, [class*="css"] {
  font-family: "DM Sans", system-ui, sans-serif;
  color: #1a1a1f;
}

.stApp {
  background: #f4f1ec;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { background: transparent; }
.block-container {
  padding-top: 0 !important;
  padding-bottom: 2rem !important;
  max-width: 980px !important;
}

.fq-hero {
  background: linear-gradient(135deg, #1f4d3f 0%, #2d6b55 55%, #3d8a6e 100%);
  color: #f8faf8;
  padding: 3.2rem 1.75rem 3.6rem;
  margin: 0 -1rem 0;
  border-radius: 0 0 18px 18px;
}
.fq-eyebrow {
  font-size: 0.8rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  opacity: 0.85;
  margin: 0 0 0.75rem;
}
.fq-hero h1 {
  font-family: "Instrument Serif", Georgia, serif;
  font-size: clamp(2.2rem, 5vw, 3.2rem);
  font-weight: 400;
  margin: 0 0 1rem;
  line-height: 1.15;
  color: #f8faf8;
}
.fq-lead {
  font-size: 1.08rem;
  max-width: 640px;
  margin: 0;
  opacity: 0.92;
  line-height: 1.55;
}

.fq-card {
  background: #ffffff;
  border: 1px solid #ddd8d0;
  border-radius: 14px;
  padding: 1.5rem 1.6rem 1.6rem;
  box-shadow: 0 8px 32px rgba(26, 26, 31, 0.08);
  margin: 1.25rem 0 0;
}
.fq-card h2 {
  font-family: "Instrument Serif", Georgia, serif;
  font-size: 1.65rem;
  font-weight: 400;
  margin: 0 0 0.45rem;
  color: #1a1a1f;
}
.fq-card h3 {
  font-size: 1.05rem;
  margin: 0 0 0.6rem;
  color: #1a1a1f;
}
.fq-muted {
  color: #5c5c66;
  font-size: 0.95rem;
  line-height: 1.55;
  margin: 0 0 1rem;
}

.fq-score {
  font-family: "Instrument Serif", Georgia, serif;
  font-size: 4rem;
  line-height: 1;
  color: #1f4d3f;
  text-align: center;
  margin: 0.4rem 0 0.2rem;
}
.fq-score-unit {
  text-align: center;
  font-size: 0.85rem;
  color: #5c5c66;
  margin-bottom: 0.9rem;
}
.fq-badge {
  display: inline-block;
  background: #e8f0ec;
  color: #2d5a4a;
  font-weight: 600;
  padding: 0.35rem 0.9rem;
  border-radius: 999px;
  font-size: 0.9rem;
  margin: 0 auto 0.55rem;
}
.fq-badge-wrap { text-align: center; }
.fq-band-summary {
  text-align: center;
  color: #5c5c66;
  font-size: 0.95rem;
  margin: 0 0 1.1rem;
}

.fq-gauge-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
  color: #5c5c66;
  margin-bottom: 0.35rem;
}
.fq-gauge-track {
  position: relative;
  height: 10px;
  background: #ece8e2;
  border-radius: 999px;
}
.fq-gauge-fill {
  height: 100%;
  background: linear-gradient(90deg, #c45c4a, #d4a84b, #3d8a6e);
  border-radius: 999px;
}
.fq-gauge-marker {
  position: absolute;
  top: 50%;
  width: 16px;
  height: 16px;
  background: #1f4d3f;
  border: 3px solid white;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}
.fq-gauge-caption {
  font-size: 0.82rem;
  color: #5c5c66;
  margin: 0.55rem 0 0;
}

.fq-info {
  background: #e8f0ec;
  border-radius: 10px;
  padding: 1rem 1.1rem;
  margin: 1rem 0;
  color: #1a1a1f;
  font-size: 0.9rem;
}
.fq-info h4 {
  margin: 0 0 0.45rem;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #5c5c66;
}
.fq-info ul { margin: 0; padding-left: 1.15rem; }
.fq-info li { margin-bottom: 0.4rem; }

.fq-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}
.fq-table th, .fq-table td {
  text-align: left;
  padding: 0.45rem 0.55rem;
  border-bottom: 1px solid #ddd8d0;
}
.fq-table th { color: #5c5c66; font-weight: 500; }

.fq-reports {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 0.85rem;
  margin-top: 0.75rem;
}
.fq-report {
  display: block;
  padding: 1rem;
  border: 1px solid #ddd8d0;
  border-radius: 10px;
  text-decoration: none !important;
  color: #1a1a1f !important;
  background: #fff;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.fq-report:hover {
  border-color: #2d5a4a;
  box-shadow: 0 4px 16px rgba(45, 90, 74, 0.12);
}
.fq-report-type {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #2d5a4a;
  font-weight: 600;
}
.fq-report strong { display: block; margin: 0.25rem 0; }
.fq-report span.desc { font-size: 0.82rem; color: #5c5c66; }

.fq-footer {
  text-align: center;
  padding: 1.5rem 0 0.5rem;
  font-size: 0.82rem;
  color: #5c5c66;
}
.fq-footer code {
  background: #e8e4de;
  padding: 0.1em 0.35em;
  border-radius: 4px;
}

div[data-testid="stFileUploader"] {
  background: #fff;
  border: 2px dashed #ddd8d0;
  border-radius: 14px;
  padding: 0.75rem;
}
div[data-testid="stFileUploader"]:hover {
  border-color: #2d5a4a;
  background: #e8f0ec;
}
</style>
"""


@st.cache_resource(show_spinner="Loading quality model…")
def load_scorer() -> QualityScorer:
    return QualityScorer(ROOT / "best_model_full.pt")


def _gauge_html(percentile: float, lo: float, hi: float) -> str:
    pct = max(0.0, min(100.0, percentile))
    return f"""
    <div class="fq-gauge-labels">
      <span>{lo:.1f}</span>
      <span>Score range on training data</span>
      <span>{hi:.1f}</span>
    </div>
    <div class="fq-gauge-track">
      <div class="fq-gauge-fill" style="width:{pct}%;"></div>
      <div class="fq-gauge-marker" style="left:{pct}%;"></div>
    </div>
    <p class="fq-gauge-caption">
      This score sits at the {pct:.0f}th percentile of the model's training range
      ({lo:.1f}–{hi:.1f}). Most FFHQ faces land near the middle (~20–26).
    </p>
    """


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="fq-hero">
          <p class="fq-eyebrow">OFIQ Unified Quality · SmallResNet CNN</p>
          <h1>Face Image Quality Scorer</h1>
          <p class="fq-lead">
            Upload a face photo and get an instant <strong>UnifiedQualityScore</strong>
            prediction — the same biometric quality metric OFIQ computes, approximated
            by a compact neural network trained on 70,000 FFHQ images.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scorer = load_scorer()

    st.markdown(
        """
        <div class="fq-card">
          <h2>Score an image</h2>
          <p class="fq-muted">
            The model was trained on <strong>tightly cropped FFHQ face portraits</strong>
            (256×256). For phone or web photos we detect the face and crop it first —
            without that crop, even sharp portraits score artificially low because the
            network “sees” mostly background.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drop a face image here or browse",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        raw = uploaded.getvalue()
        original = Image.open(io.BytesIO(raw)).convert("RGB")

        with st.spinner("Detecting face and scoring…"):
            try:
                result = scorer.score_image_bytes(raw)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not score image: {exc}")
                return

        crop = Image.open(io.BytesIO(base64.b64decode(result.crop_preview_b64)))

        st.markdown(
            '<div class="fq-card"><h2>Your quality score</h2></div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns([1, 1.15], gap="large")

        with left:
            st.markdown(
                f"""
                <div style="text-align:center;">
                  <div class="fq-score">{result.score:.2f}</div>
                  <div class="fq-score-unit">native OFIQ units</div>
                  <div class="fq-badge-wrap"><span class="fq-badge">{result.band}</span></div>
                  <p class="fq-band-summary">{result.band_summary}</p>
                </div>
                {_gauge_html(result.percentile, result.lo, result.hi)}
                """,
                unsafe_allow_html=True,
            )
            st.markdown("##### Face crop sent to the model")
            st.image(crop, width=180)
            st.caption(result.crop_note)
            st.markdown("##### Your upload")
            st.image(original, use_container_width=True)

        with right:
            st.markdown(
                f"""
                <h3>What this score means</h3>
                <p class="fq-muted">{result.band_detail}</p>
                <div class="fq-info">
                  <h4>How to read the number</h4>
                  <ul>
                    <li><strong>Higher = better biometric quality</strong> — sharpness, pose, lighting, occlusion — not beauty.</li>
                    <li><strong>Most faces score ~20–26.</strong> That is normal, not failure. Scores above ~28 are uncommon; above 30 is rare (~1%).</li>
                    <li><strong>Scale is ~{result.lo:.0f}–{result.hi:.0f} native units</strong>, not 0–100.</li>
                    <li><strong>Approximate, not exact OFIQ.</strong> Average error ≈ ±{result.mae:.2f} on held-out FFHQ (Pearson <em>r</em> ≈ 0.87).</li>
                    <li><strong>Crop matters.</strong> Check the face crop on the left — if it’s wrong, the score will be wrong.</li>
                  </ul>
                </div>
                <h4 style="color:#5c5c66;text-transform:uppercase;letter-spacing:0.06em;font-size:0.8rem;">Quality bands (heuristic)</h4>
                <table class="fq-table">
                  <thead><tr><th>Score</th><th>Band</th><th>Typical meaning</th></tr></thead>
                  <tbody>
                    <tr><td>&lt; 16</td><td>Very low</td><td>Serious usability issues</td></tr>
                    <tr><td>16 – 20</td><td>Below average</td><td>Moderate limitations</td></tr>
                    <tr><td>20 – 26</td><td>Average</td><td>Typical, generally usable</td></tr>
                    <tr><td>26 – 29</td><td>Above average</td><td>Good biometric quality</td></tr>
                    <tr><td>&gt; 29</td><td>High</td><td>Among the best in dataset</td></tr>
                  </tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div class="fq-card">
          <h2>About this project</h2>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;">
            <div>
              <h3>The problem</h3>
              <p class="fq-muted">
                <a href="https://www.bsi.bund.de/OFIQ" target="_blank" rel="noopener">OFIQ</a>
                measures how suitable a face image is for biometric matching, but it is
                slow (seconds per image). This <strong>SmallResNet</strong> (~0.33M params)
                predicts <code>UnifiedQualityScore.native</code> in milliseconds
                (MAE 1.32, Spearman ρ ≈ 0.87 on 7k held-out faces).
              </p>
            </div>
            <div>
              <h3>Why it matters</h3>
              <p class="fq-muted">
                Fast quality estimation enables large-scale filtering of face datasets
                and can serve as a critic inside generative pipelines — without running
                full OFIQ on every sample.
              </p>
            </div>
          </div>
        </div>

        <div class="fq-card">
          <h2>Technical reports &amp; documentation</h2>
          <p class="fq-muted">Full methodology, evaluation metrics, and known limitations:</p>
          <div class="fq-reports">
            <a class="fq-report" href="{GITHUB}/blob/main/REPORT.md" target="_blank" rel="noopener">
              <span class="fq-report-type">Markdown</span>
              <strong>Project Report</strong>
              <span class="desc">REPORT.md — dataset, architecture, training results</span>
            </a>
            <a class="fq-report" href="{GITHUB}/blob/main/Technical%20Reports/tech_report.pdf" target="_blank" rel="noopener">
              <span class="fq-report-type">PDF</span>
              <strong>Technical Report</strong>
              <span class="desc">Academic write-up with figures and methodology</span>
            </a>
            <a class="fq-report" href="{GITHUB}/blob/main/Technical%20Reports/fixes/fixes.pdf" target="_blank" rel="noopener">
              <span class="fq-report-type">PDF</span>
              <strong>Problems &amp; Fixes</strong>
              <span class="desc">Known issues and how they were addressed</span>
            </a>
            <a class="fq-report" href="{GITHUB}" target="_blank" rel="noopener">
              <span class="fq-report-type">GitHub</span>
              <strong>Source code</strong>
              <span class="desc">Training code, weights, and evaluation scripts</span>
            </a>
          </div>
        </div>

        <p class="fq-footer">
          Checkpoint: <code>best_model_full.pt</code> (SmallResNet, epoch {scorer.epoch})
          · Trained on <code>UnifiedQualityScore.native</code>
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
