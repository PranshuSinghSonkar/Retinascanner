"""Inference and reporting services used by Flask routes."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


CLASSIFICATIONS = {
    0: {
        "name": "No Diabetic Retinopathy",
        "severity": "Low risk",
        "badge": "success",
        "description": "No visible signs of diabetic retinopathy were identified by the model.",
        "clinical_explanation": "The retinal image appears consistent with no detectable diabetic retinal changes.",
        "next_step": "Continue routine diabetes and eye-care follow-up as advised by your clinician.",
    },
    1: {
        "name": "Mild Diabetic Retinopathy",
        "severity": "Mild",
        "badge": "info",
        "description": "Early retinal changes associated with diabetic retinopathy were detected.",
        "clinical_explanation": "Small retinal vessel changes can occur before vision symptoms are noticeable.",
        "next_step": "Arrange a non-urgent review with an eye-care professional for confirmation.",
    },
    2: {
        "name": "Moderate Diabetic Retinopathy",
        "severity": "Moderate",
        "badge": "warning",
        "description": "The model identified retinal features that may be consistent with moderate disease.",
        "clinical_explanation": "Moderate diabetic retinal changes may require closer monitoring and clinical assessment.",
        "next_step": "Schedule an ophthalmology assessment to discuss monitoring and treatment options.",
    },
    3: {
        "name": "Severe Diabetic Retinopathy",
        "severity": "High risk",
        "badge": "danger",
        "description": "The model identified features associated with severe diabetic retinopathy.",
        "clinical_explanation": "Advanced retinal vascular changes can increase the risk of sight-threatening complications.",
        "next_step": "Seek a prompt ophthalmology appointment for a comprehensive clinical examination.",
    },
    4: {
        "name": "Proliferative Diabetic Retinopathy",
        "severity": "Urgent review",
        "badge": "danger",
        "description": "The model identified features that may be associated with proliferative disease.",
        "clinical_explanation": "This category can involve abnormal blood-vessel growth and requires specialist evaluation.",
        "next_step": "Contact an ophthalmologist urgently for an in-person assessment.",
    },
}


class PredictionService:
    """Lazy-loading service that isolates model inference from Flask routes."""

    def __init__(self, models_dir: Path) -> None:
        self.model_path = models_dir / "retina_model.keras"
        self._model = None

    def _load_model(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            import tensorflow as tf

            self._model = tf.keras.models.load_model(self.model_path)
        return self._model

    @staticmethod
    def _prepare_image(image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("This file is not a readable retinal image.")
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # The model was trained with EfficientNet's ImageNet input convention.
        return np.expand_dims(image * 255.0, axis=0)

    def analyze(self, image_path: Path) -> dict[str, object]:
        """Run one image through the saved classifier and return presentation data."""
        model = self._load_model()
        probabilities = model.predict(self._prepare_image(image_path), verbose=0)[0]
        predicted_class = int(np.argmax(probabilities))
        details = CLASSIFICATIONS[predicted_class].copy()
        details.update(
            {
                "confidence": round(float(probabilities[predicted_class]) * 100, 1),
                "predicted_class": predicted_class,
                "probabilities": [round(float(value) * 100, 1) for value in probabilities],
                "labels": [CLASSIFICATIONS[index]["name"] for index in range(5)],
            }
        )
        return details

    @staticmethod
    def create_pdf_report(result: dict[str, object]) -> BytesIO:
        """Generate a concise patient-facing PDF report from a completed result."""
        report = BytesIO()
        document = SimpleDocTemplate(report, pagesize=A4, rightMargin=48, leftMargin=48)
        styles = getSampleStyleSheet()
        content = [
            Paragraph("RetinaAI Analysis Report", styles["Title"]),
            Spacer(1, 0.2 * inch),
            Paragraph(f"<b>Prediction:</b> {result['name']}", styles["BodyText"]),
            Paragraph(f"<b>Confidence:</b> {result['confidence']}%", styles["BodyText"]),
            Paragraph(f"<b>Severity:</b> {result['severity']}", styles["BodyText"]),
            Spacer(1, 0.15 * inch),
            Paragraph(f"<b>Clinical explanation:</b> {result['clinical_explanation']}", styles["BodyText"]),
            Spacer(1, 0.1 * inch),
            Paragraph(f"<b>Recommended next step:</b> {result['next_step']}", styles["BodyText"]),
            Spacer(1, 0.2 * inch),
        ]
        table_rows = [["Class", "Probability"]] + [
            [label, f"{probability}%"] for label, probability in zip(result["labels"], result["probabilities"])
        ]
        table = Table(table_rows, colWidths=[4.2 * inch, 1.2 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F4C81")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D6E1EC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))
        content.extend([table, Spacer(1, 0.25 * inch), Paragraph(
            "Medical disclaimer: RetinaAI is an AI decision-support tool and does not replace diagnosis by a qualified clinician.",
            styles["BodyText"],
        )])
        document.build(content)
        report.seek(0)
        return report
