#!/usr/bin/env python3
"""Face image quality scoring demo — Flask app."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from inference import QualityScorer

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "Technical Reports"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

GITHUB_REPO = "https://github.com/inezaodon/image-quality-cnn"

scorer: QualityScorer | None = None


def get_scorer() -> QualityScorer:
    global scorer
    if scorer is None:
        scorer = QualityScorer()
    return scorer


@app.route("/")
def index():
    return render_template(
        "index.html",
        github_repo=GITHUB_REPO,
        report_links={
            "project_report_md": f"{GITHUB_REPO}/blob/main/REPORT.md",
            "project_report_raw": f"{GITHUB_REPO}/raw/main/REPORT.md",
            "tech_report_pdf": "/reports/tech_report.pdf",
            "fixes_pdf": "/reports/fixes.pdf",
            "fixes_md": f"{GITHUB_REPO}/blob/main/Technical%20Reports/fixes/FIXES.md",
            "graph_explanation": f"{GITHUB_REPO}/blob/main/graph_explanation.txt",
        },
    )


@app.route("/api/score", methods=["POST"])
def api_score():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    try:
        result = get_scorer().score_image_bytes(file.read())
    except Exception as exc:  # noqa: BLE001 — surface inference errors to client
        return jsonify({"error": f"Could not score image: {exc}"}), 400

    return jsonify(dataclasses.asdict(result))


@app.route("/reports/<path:filename>")
def serve_report(filename: str):
    allowed = {
        "tech_report.pdf": REPORTS / "tech_report.pdf",
        "fixes.pdf": REPORTS / "fixes" / "fixes.pdf",
        "REPORT.md": ROOT / "REPORT.md",
        "graph_explanation.txt": ROOT / "graph_explanation.txt",
    }
    if filename not in allowed:
        return "Not found", 404
    path = allowed[filename]
    return send_from_directory(path.parent, path.name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    get_scorer()
    print(f"Model loaded. Open http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True)
