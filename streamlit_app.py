"""Streamlit demo matching the Flask web UI — scoring + in-app PDF viewing."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

from web.inference import QualityScorer

GITHUB = "https://github.com/inezaodon/image-quality-cnn"
ROOT = Path(__file__).resolve().parent

REPORT_FILES = {
    "project_md": ROOT / "REPORT.md",
    "tech_pdf": ROOT / "Technical Reports" / "tech_report.pdf",
    "fixes_pdf": ROOT / "Technical Reports" / "fixes" / "fixes.pdf",
    "fixes_md": ROOT / "Technical Reports" / "fixes" / "FIXES.md",
    "graphs": ROOT / "graph_explanation.txt",
}

st.set_page_config(
    page_title="Face Image Quality Scorer",
    page_icon="◉",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=Instrument+Serif:ital@0;1&display=swap');

html, body, [class*="css"]  {
  font-family: "DM Sans", system-ui, sans-serif !important;
  color: #1a1a1f;
}
.stApp { background: #f4f1ec; }
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"] {
  visibility: hidden; display: none;
}
[data-testid="stHeader"] { background: transparent; }
.block-container {
  padding-top: 0 !important;
  padding-bottom: 2.5rem !important;
  max-width: 980px !important;
}

/* Hero */
.fq-hero {
  background: linear-gradient(135deg, #1f4d3f 0%, #2d6b55 55%, #3d8a6e 100%);
  color: #f8faf8;
  padding: 3.2rem 1.75rem 3.8rem;
  margin: 0 -1.5rem 1.5rem;
  border-radius: 0 0 18px 18px;
}
.fq-eyebrow {
  font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase;
  opacity: 0.85; margin: 0 0 0.75rem;
}
.fq-hero h1 {
  font-family: "Instrument Serif", Georgia, serif !important;
  font-size: clamp(2.1rem, 5vw, 3.1rem); font-weight: 400;
  margin: 0 0 1rem; line-height: 1.15; color: #f8faf8 !important;
}
.fq-lead { font-size: 1.05rem; max-width: 640px; margin: 0; opacity: 0.93; line-height: 1.55; }

/* Cards */
.fq-card {
  background: #ffffff; border: 1px solid #ddd8d0; border-radius: 14px;
  padding: 1.5rem 1.6rem; box-shadow: 0 8px 32px rgba(26,26,31,0.08);
  margin: 0 0 1.25rem;
}
.fq-card h2 {
  font-family: "Instrument Serif", Georgia, serif !important;
  font-size: 1.6rem; font-weight: 400; margin: 0 0 0.4rem; color: #1a1a1f !important;
}
.fq-card h3 { font-size: 1.05rem; margin: 0 0 0.55rem; color: #1a1a1f !important; }
.fq-muted { color: #5c5c66; font-size: 0.95rem; line-height: 1.55; margin: 0 0 0.85rem; }

/* Score */
.fq-score {
  font-family: "Instrument Serif", Georgia, serif !important;
  font-size: 4rem; line-height: 1; color: #1f4d3f; text-align: center; margin: 0.35rem 0 0.15rem;
}
.fq-score-unit { text-align: center; font-size: 0.85rem; color: #5c5c66; margin-bottom: 0.85rem; }
.fq-badge-wrap { text-align: center; }
.fq-badge {
  display: inline-block; background: #e8f0ec; color: #2d5a4a; font-weight: 600;
  padding: 0.35rem 0.9rem; border-radius: 999px; font-size: 0.9rem;
}
.fq-band-summary { text-align: center; color: #5c5c66; font-size: 0.95rem; margin: 0.55rem 0 1rem; }

.fq-gauge-labels {
  display: flex; justify-content: space-between; font-size: 0.75rem; color: #5c5c66; margin-bottom: 0.35rem;
}
.fq-gauge-track { position: relative; height: 10px; background: #ece8e2; border-radius: 999px; }
.fq-gauge-fill {
  height: 100%; background: linear-gradient(90deg, #c45c4a, #d4a84b, #3d8a6e); border-radius: 999px;
}
.fq-gauge-marker {
  position: absolute; top: 50%; width: 16px; height: 16px; background: #1f4d3f;
  border: 3px solid white; border-radius: 50%; transform: translate(-50%, -50%);
  box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}
.fq-gauge-caption { font-size: 0.82rem; color: #5c5c66; margin: 0.55rem 0 0; }

.fq-info {
  background: #e8f0ec; border-radius: 10px; padding: 1rem 1.1rem; margin: 0.85rem 0 1rem;
  color: #1a1a1f; font-size: 0.9rem;
}
.fq-info h4 {
  margin: 0 0 0.45rem; font-size: 0.78rem; text-transform: uppercase;
  letter-spacing: 0.06em; color: #5c5c66;
}
.fq-info ul { margin: 0; padding-left: 1.15rem; }
.fq-info li { margin-bottom: 0.4rem; }

.fq-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.fq-table th, .fq-table td {
  text-align: left; padding: 0.45rem 0.55rem; border-bottom: 1px solid #ddd8d0;
}
.fq-table th { color: #5c5c66; font-weight: 500; }

.fq-img-frame {
  border: 1px solid #ddd8d0; border-radius: 10px; overflow: hidden;
  background: #f0eeea; padding: 0.35rem; margin-bottom: 0.35rem;
}
.fq-img-label {
  font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: #5c5c66; margin: 0.75rem 0 0.35rem;
}

.fq-report-type {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
  color: #2d5a4a; font-weight: 600;
}
.fq-footer {
  text-align: center; padding: 1.25rem 0 0.25rem; font-size: 0.82rem; color: #5c5c66;
}
.fq-footer code { background: #e8e4de; padding: 0.1em 0.35em; border-radius: 4px; }

/* Streamlit widgets */
div[data-testid="stFileUploader"] {
  background: #fff; border: 2px dashed #ddd8d0; border-radius: 14px; padding: 0.85rem;
}
div[data-testid="stFileUploader"]:hover { border-color: #2d5a4a; background: #e8f0ec; }
div[data-testid="stImage"] img {
  border-radius: 10px;
}
.stDownloadButton button, .stButton > button {
  border-radius: 999px !important;
  border: 1px solid #2d5a4a !important;
  color: #2d5a4a !important;
  background: #fff !important;
  font-weight: 600 !important;
}
.stDownloadButton button:hover, .stButton > button:hover {
  background: #e8f0ec !important;
  border-color: #1f4d3f !important;
}
button[kind="primary"] {
  background: #2d5a4a !important; color: #fff !important; border: none !important;
}
</style>
"""


@st.cache_resource(show_spinner="Loading quality model…")
def load_scorer() -> QualityScorer:
    return QualityScorer(ROOT / "best_model_full.pt")


@st.cache_data(show_spinner=False)
def _file_bytes(path_str: str) -> bytes:
    return Path(path_str).read_bytes()


def _gauge_html(percentile: float, lo: float, hi: float) -> str:
    pct = max(0.0, min(100.0, percentile))
    return f"""
    <div class="fq-gauge-labels">
      <span>{lo:.1f}</span><span>Score range on training data</span><span>{hi:.1f}</span>
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


def _show_pdf_inline(path: Path, height: int = 720) -> None:
    """Show a PDF inside the app from local repo bytes (no GitHub viewer)."""
    if not path.exists():
        st.error(f"PDF not found: {path.name}")
        return

    # Preferred: native Streamlit PDF viewer (streamlit-pdf).
    try:
        st.pdf(path, height=height)
        return
    except Exception:
        pass

    # Fallback: embed local bytes as a data-URI (works without extra packages).
    data = _file_bytes(str(path))
    b64 = base64.b64encode(data).decode("ascii")
    components.html(
        f"""
        <div style="border:1px solid #ddd8d0;border-radius:12px;overflow:hidden;background:#f7f5f1;">
          <object data="data:application/pdf;base64,{b64}" type="application/pdf"
                  width="100%" height="{height}px">
            <p style="padding:1rem;font-family:system-ui;color:#5c5c66;">
              Your browser could not display this PDF inline.
              Use the download button above instead.
            </p>
          </object>
        </div>
        """,
        height=height + 16,
        scrolling=True,
    )


def _pil_preview(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")


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

    # ---- Upload ----------------------------------------------------------- #
    st.markdown(
        """
        <div class="fq-card">
          <h2>Score an image</h2>
          <p class="fq-muted">
            The model was trained on <strong>tightly cropped FFHQ face portraits</strong>
            (256×256). For phone or web photos we detect the face and crop it first —
            without that crop, even sharp portraits score artificially low.
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
        original = _pil_preview(raw)

        with st.spinner("Detecting face and scoring…"):
            try:
                result = scorer.score_image_bytes(raw)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not score image: {exc}")
                return

        crop = Image.open(io.BytesIO(base64.b64decode(result.crop_preview_b64)))

        prev_l, prev_r = st.columns([1, 1], gap="medium")
        with prev_l:
            st.markdown('<p class="fq-img-label">Your upload</p>', unsafe_allow_html=True)
            st.image(original, use_container_width=True)
        with prev_r:
            st.markdown(
                '<p class="fq-img-label">Face crop sent to the model</p>',
                unsafe_allow_html=True,
            )
            st.image(crop, use_container_width=True)
            st.caption(result.crop_note)

        # ---- Results ------------------------------------------------------ #
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
        with right:
            st.markdown(
                f"""
                <h3>What this score means</h3>
                <p class="fq-muted">{result.band_detail}</p>
                <div class="fq-info">
                  <h4>How to read the number</h4>
                  <ul>
                    <li><strong>Higher = better biometric quality</strong> — sharpness, pose, lighting, occlusion — not beauty.</li>
                    <li><strong>Most faces score ~20–26.</strong> That is normal. Above ~28 is uncommon; above 30 is rare (~1%).</li>
                    <li><strong>Scale is ~{result.lo:.0f}–{result.hi:.0f}</strong> native units, not 0–100.</li>
                    <li><strong>Approximate, not exact OFIQ.</strong> Average error ≈ ±{result.mae:.2f} (Pearson <em>r</em> ≈ 0.87).</li>
                    <li><strong>Crop matters.</strong> If the crop above is wrong, the score will be wrong.</li>
                  </ul>
                </div>
                <h4 style="color:#5c5c66;text-transform:uppercase;letter-spacing:0.06em;font-size:0.78rem;">
                  Quality bands (heuristic)
                </h4>
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

    # ---- About ------------------------------------------------------------ #
    st.markdown(
        """
        <div class="fq-card">
          <h2>About this project</h2>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;">
            <div>
              <h3>The problem</h3>
              <p class="fq-muted">
                <a href="https://www.bsi.bund.de/OFIQ" target="_blank" rel="noopener">OFIQ</a>
                measures how suitable a face image is for biometric matching, but it is
                slow. This <strong>SmallResNet</strong> (~0.33M params) predicts
                <code>UnifiedQualityScore.native</code> in milliseconds
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
        """,
        unsafe_allow_html=True,
    )

    # ---- Reports (in-app viewer — no GitHub PDF pages) -------------------- #
    st.markdown(
        """
        <div class="fq-card">
          <h2>Technical reports &amp; documentation</h2>
          <p class="fq-muted">
            Open a document below to view it <strong>inside this app</strong>
            (GitHub does not display PDFs). You can also download any file.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    choice = st.radio(
        "Choose a document",
        [
            "Technical Report (PDF)",
            "Problems & Fixes (PDF)",
            "Project Report (REPORT.md)",
            "Fixes source (FIXES.md)",
            "Understanding the graphs",
            "None — hide viewer",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<span class="fq-report-type">PDF</span>', unsafe_allow_html=True)
        st.download_button(
            "Download tech report",
            data=_file_bytes(str(REPORT_FILES["tech_pdf"])) if REPORT_FILES["tech_pdf"].exists() else b"",
            file_name="tech_report.pdf",
            mime="application/pdf",
            key="dl_tech",
            use_container_width=True,
            disabled=not REPORT_FILES["tech_pdf"].exists(),
        )
    with c2:
        st.markdown('<span class="fq-report-type">PDF</span>', unsafe_allow_html=True)
        st.download_button(
            "Download fixes PDF",
            data=_file_bytes(str(REPORT_FILES["fixes_pdf"])) if REPORT_FILES["fixes_pdf"].exists() else b"",
            file_name="fixes.pdf",
            mime="application/pdf",
            key="dl_fixes",
            use_container_width=True,
            disabled=not REPORT_FILES["fixes_pdf"].exists(),
        )
    with c3:
        st.markdown('<span class="fq-report-type">Markdown</span>', unsafe_allow_html=True)
        st.download_button(
            "Download REPORT.md",
            data=_file_bytes(str(REPORT_FILES["project_md"])) if REPORT_FILES["project_md"].exists() else b"",
            file_name="REPORT.md",
            mime="text/markdown",
            key="dl_report",
            use_container_width=True,
            disabled=not REPORT_FILES["project_md"].exists(),
        )
    with c4:
        st.markdown('<span class="fq-report-type">GitHub</span>', unsafe_allow_html=True)
        st.link_button("Open source repo", GITHUB, use_container_width=True)

    if choice == "Technical Report (PDF)":
        st.markdown("##### Technical Report")
        _show_pdf_inline(REPORT_FILES["tech_pdf"], height=780)
    elif choice == "Problems & Fixes (PDF)":
        st.markdown("##### Problems & Fixes")
        _show_pdf_inline(REPORT_FILES["fixes_pdf"], height=780)
    elif choice == "Project Report (REPORT.md)":
        st.markdown("##### Project Report")
        if REPORT_FILES["project_md"].exists():
            st.markdown(REPORT_FILES["project_md"].read_text(encoding="utf-8"))
        else:
            st.error("REPORT.md not found")
    elif choice == "Fixes source (FIXES.md)":
        st.markdown("##### Fixes (source)")
        if REPORT_FILES["fixes_md"].exists():
            st.markdown(REPORT_FILES["fixes_md"].read_text(encoding="utf-8"))
        else:
            st.error("FIXES.md not found")
    elif choice == "Understanding the graphs":
        st.markdown("##### Understanding the graphs")
        if REPORT_FILES["graphs"].exists():
            st.text(REPORT_FILES["graphs"].read_text(encoding="utf-8"))
        else:
            st.error("graph_explanation.txt not found")

    st.markdown(
        f"""
        <p class="fq-footer">
          Checkpoint: <code>best_model_full.pt</code> (SmallResNet, epoch {scorer.epoch})
          · Trained on <code>UnifiedQualityScore.native</code>
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
