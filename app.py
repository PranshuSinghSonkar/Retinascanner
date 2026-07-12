"""Flask web application entry point for RetinaAI."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from flask import Flask, abort, flash, redirect, render_template, request, send_file, session, url_for
from werkzeug.utils import secure_filename

from src.prediction.service import PredictionService


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app() -> Flask:
    """Create and configure the RetinaAI Flask application."""
    import sys

    _model_path = Path(__file__).resolve().parent / "models" / "retina_model.keras"

    print("RetinaAI Starting...")
    if _model_path.exists():
        print(f"Model found: {_model_path}")
    else:
        raise RuntimeError(
            "Production model not found: models/retina_model.keras"
        )

    import tensorflow as tf
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")

    app = Flask(__name__)
    app.config.from_object("config.Config")
    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
    prediction_service = PredictionService(
        Path(app.config["MODEL_DIR"]),
        Path(app.config["UPLOAD_DIR"]),
    )

    @app.get("/")
    def home() -> str:
        return render_template("home.html")

    @app.get("/about")
    def about() -> str:
        return render_template("about.html")

    @app.route("/prediction", methods=["GET", "POST"])
    def prediction() -> str:
        if request.method == "GET":
            return render_template("prediction.html")

        upload = request.files.get("retina_image")
        if upload is None or not upload.filename:
            flash("Please select a retinal fundus image to continue.", "danger")
            return redirect(url_for("prediction"))
        if not _allowed_file(upload.filename):
            flash("Use a PNG, JPG, or JPEG image under 10 MB.", "danger")
            return redirect(url_for("prediction"))

        filename = f"{uuid4().hex}_{secure_filename(upload.filename)}"
        upload_path = Path(app.config["UPLOAD_DIR"]) / filename
        upload.save(upload_path)
        try:
            result = prediction_service.analyze(upload_path)
        except ValueError as error:
            upload_path.unlink(missing_ok=True)
            flash(str(error), "danger")
            return redirect(url_for("prediction"))
        except Exception:
            upload_path.unlink(missing_ok=True)
            app.logger.exception("RetinaAI prediction failed")
            flash("We could not complete the analysis. Please try again shortly.", "danger")
            return redirect(url_for("prediction"))

        result["image_url"] = url_for("static", filename=f"uploads/{filename}")
        if result.get("heatmap_path"):
            heatmap_filename = Path(str(result["heatmap_path"])).name
            result["heatmap_url"] = url_for("static", filename=f"uploads/heatmaps/{heatmap_filename}")
        session["prediction_result"] = result
        return redirect(url_for("results"))

    @app.get("/results")
    def results() -> str:
        result = session.get("prediction_result")
        if result is None:
            flash("Upload an image to view an analysis report.", "info")
            return redirect(url_for("prediction"))
        return render_template("results.html", result=result)

    @app.get("/report/download")
    def download_report():
        result = session.get("prediction_result")
        if result is None:
            abort(404)
        report = prediction_service.create_pdf_report(result)
        return send_file(
            report,
            as_attachment=True,
            download_name="retinaai-report.pdf",
            mimetype="application/pdf",
        )

    @app.errorhandler(404)
    def not_found(_: object):
        return render_template("404.html"), 404

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
