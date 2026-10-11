"""One process: a JSON API for machines and a review screen for the inspector.

    python -m defects.app            # http://127.0.0.1:7860

    GET  /api/health                 model, classes and thresholds in use
    POST /api/detect                 multipart image -> detections and verdict
    GET  /api/reviews/summary        how often the inspector agreed with the model
    /                                the review screen
"""

from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from . import __version__, reviews
from .config import SAMPLES_DIR
from .service import Inspection, Inspector, draw

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
VERDICT_STYLE = {"pass": ("#1e8e3e", "PASS"), "review": ("#b06000", "REVIEW"), "reject": ("#c5221f", "REJECT")}


@lru_cache(maxsize=1)
def get_inspector() -> Inspector:
    # Loaded on first use, not at import, so tests can import the app cheaply
    # and a missing model file produces a clear error on the first request.
    return Inspector()


def decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise HTTPException(400, "The upload is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The image is larger than 10 MB.")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(415, "The file is not an image this service can read (JPEG, PNG, BMP or TIFF).")
    return image


def create_api() -> FastAPI:
    api = FastAPI(title="Steel surface defect detection", version=__version__)

    @api.get("/api/health")
    def health() -> dict:
        inspector = get_inspector()
        t = inspector.thresholds
        return {
            "status": "ok",
            "version": __version__,
            "model": inspector.detector.path.name,
            "runtime": "onnxruntime-cpu",
            "input_size": inspector.detector.size,
            "classes": list(inspector.detector.class_names),
            "thresholds": {**dict(zip(inspector.detector.class_names, t.per_class)), "review_floor": t.review_floor},
        }

    @api.post("/api/detect")
    async def detect(file: UploadFile = File(...), sensitivity: float = Form(1.0)) -> dict:
        if not 0.25 <= sensitivity <= 4.0:
            raise HTTPException(422, "sensitivity must be between 0.25 and 4.0")
        image = decode_image(await file.read())
        return {"file": file.filename, **get_inspector().inspect(image, sensitivity).to_dict()}

    @api.get("/api/reviews/summary")
    def review_summary() -> dict:
        return reviews.summary()

    return api


def verdict_html(inspection: Inspection) -> str:
    colour, text = VERDICT_STYLE[inspection.verdict]
    # Colours are set on every element with !important because the Gradio
    # theme otherwise repaints the text dark, which is unreadable on red.
    white = "color:#fff !important"
    return (
        f'<div style="padding:14px 18px;border-radius:8px;background:{colour}">'
        f'<div style="{white};font-size:26px;font-weight:700;letter-spacing:1px">{text}</div>'
        f'<div style="{white};font-size:15px;margin-top:4px">{inspection.reason}</div>'
        f'<div style="{white};font-size:12px;margin-top:6px;opacity:.85">{inspection.latency_ms:.0f} ms on CPU</div></div>'
    )


def build_screen():
    import gradio as gr

    def run(image_rgb: np.ndarray | None, sensitivity: float):
        if image_rgb is None:
            raise gr.Error("Upload an image or pick one of the examples first.")
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        inspection = get_inspector().inspect(image, sensitivity)
        rows = [
            [d.label, round(d.confidence, 2), "yes" if sure else "needs review", round(d.x2 - d.x1), round(d.y2 - d.y1)]
            for d, sure in zip(inspection.detections, inspection.confident)
        ]
        annotated = cv2.cvtColor(draw(image, inspection), cv2.COLOR_BGR2RGB)
        return annotated, verdict_html(inspection), rows, (image, inspection), ""

    def save(state, judgement: str, note: str) -> str:
        if state is None:
            raise gr.Error("Inspect an image before saving a review.")
        image, inspection = state
        reviews.save_review(image, inspection, judgement, note or "")
        s = reviews.summary()
        plural = "review" if s["reviews"] == 1 else "reviews"
        return f"Saved. {s['reviews']} {plural} so far; the inspector agreed with the model in {s['by_judgement']['correct']}."

    examples = sorted(str(p) for p in SAMPLES_DIR.glob("*.jpg"))
    with gr.Blocks(title="Steel surface defect inspection", analytics_enabled=False) as screen:
        gr.Markdown(
            "# Steel surface defect inspection\n"
            "Upload a photo of the surface. The model marks every defect it finds, names its type and proposes a verdict. "
            "Thick boxes are confident detections; thin boxes with a question mark are below the threshold and need a person."
        )
        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="numpy", label="Surface image", height=320)
                sensitivity = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Threshold multiplier", info="Below 1 the model flags more (fewer missed defects, more false alarms). Above 1 it flags less.")
                inspect_btn = gr.Button("Inspect", variant="primary")
                gr.Examples(examples, inputs=image_in, label="Examples from the test set", examples_per_page=12)
            with gr.Column(scale=1):
                verdict = gr.HTML()
                image_out = gr.Image(label="Detections", height=420, interactive=False)
                table = gr.Dataframe(headers=["Defect type", "Confidence", "Confident", "Width px", "Height px"], label="Detections", interactive=False)
                gr.Markdown("### Inspector review")
                judgement = gr.Radio(list(reviews.JUDGEMENTS), value="correct", label="Is the result right?")
                note = gr.Textbox(label="Note (optional)", lines=1)
                save_btn = gr.Button("Save review")
                saved = gr.Markdown()
        inspect_btn.click(run, [image_in, sensitivity], [image_out, verdict, table, state, saved])
        save_btn.click(save, [state, judgement, note], saved)
    return screen


def create_app() -> FastAPI:
    import gradio as gr

    return gr.mount_gradio_app(create_api(), build_screen(), path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "7860")), log_level="warning")
