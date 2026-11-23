"""Streamlit frontend for the mineral classification capstone."""

from __future__ import annotations

import base64
import os
from typing import Any

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError


DEFAULT_API_URL = "http://127.0.0.1:8000"
SUPPORTED_MINERALS = (
    "Azurite",
    "Baryte",
    "Beryl",
    "Calcite",
    "Cerussite",
    "Copper",
    "Fluorite",
    "Gypsum",
    "Hematite",
    "Malachite",
    "Pyrite",
    "Pyromorphite",
    "Quartz",
    "Smithsonite",
    "Wulfenite",
)


def api_url() -> str:
    """Return the configured prediction-service URL."""
    return os.getenv("MINERAL_API_URL", DEFAULT_API_URL).rstrip("/")


def encode_image(image_bytes: bytes) -> str:
    """Encode image bytes as plain base64 text for the prediction API."""
    return base64.b64encode(image_bytes).decode("utf-8")


def request_prediction(image_bytes: bytes) -> dict[str, Any]:
    """Send one uploaded image to the remote prediction service."""
    response = requests.post(
        f"{api_url()}/predict",
        json={"uploaded_image": encode_image(image_bytes)},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def normalize_confidence(value: Any) -> float | None:
    """Return confidence as a 0-1 value when the API provides one."""
    if value is None:
        return None

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None

    if confidence > 1:
        confidence /= 100
    return max(0.0, min(confidence, 1.0))


def parse_prediction(
    data: dict[str, Any],
) -> tuple[str | None, float | None, list[dict[str, Any]]]:
    """Accept the planned response names while the API is still evolving."""
    mineral = data.get("predicted_mineral") or data.get("prediction") or data.get("mineral")
    confidence = normalize_confidence(data.get("confidence"))
    candidates = data.get("top_predictions") or data.get("predictions") or []

    if isinstance(mineral, dict):
        confidence = normalize_confidence(mineral.get("confidence", confidence))
        mineral = mineral.get("mineral") or mineral.get("label") or mineral.get("class")

    if not isinstance(candidates, list):
        candidates = []

    return mineral, confidence, candidates


def show_prediction(data: dict[str, Any]) -> None:
    """Display the prediction returned by the API."""
    mineral, confidence, candidates = parse_prediction(data)

    if mineral is None:
        st.warning("The prediction service responded, but its result format is not recognized yet.")
        with st.expander("View API response"):
            st.json(data)
        return

    st.success("Prediction completed")
    result_column, confidence_column = st.columns(2)
    result_column.metric("Predicted mineral", str(mineral).title())
    confidence_column.metric(
        "Confidence",
        f"{confidence:.1%}" if confidence is not None else "Not provided",
    )

    if candidates:
        st.subheader("Top predictions")
        for index, candidate in enumerate(candidates[:3], start=1):
            if not isinstance(candidate, dict):
                continue
            label = candidate.get("mineral") or candidate.get("label") or candidate.get("class")
            score = normalize_confidence(candidate.get("confidence") or candidate.get("probability"))
            if label:
                score_text = f" - {score:.1%}" if score is not None else ""
                st.write(f"{index}. **{str(label).title()}**{score_text}")

    st.caption(
        "Treat this result as a preliminary suggestion and confirm it with physical "
        "tests or an expert."
    )


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(
        page_title="AI-Assisted Mineral Identification",
        page_icon="🔎",
        layout="centered",
    )

    st.title("AI-Assisted Mineral Identification")
    st.write(
        "Upload a clear photograph of one mineral specimen to receive a preliminary "
        "classification from the selected EfficientNet-B0 model."
    )

    with st.sidebar:
        st.header("About")
        st.write(
            "This UPRM capstone explores convolutional neural networks for mineral-image "
            "classification. The tool is educational and is not a substitute for geological testing."
        )
        st.divider()
        with st.expander("Supported minerals"):
            for mineral in SUPPORTED_MINERALS:
                st.write(f"- {mineral}")

    st.subheader("Classify a mineral")
    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image",
        type=("jpg", "jpeg", "png"),
        help="For the clearest result, use one centered specimen, even lighting, and a simple background.",
    )

    if uploaded_file is None:
        st.info("Upload an image to begin.")
        return

    image_bytes = uploaded_file.getvalue()
    try:
        image = Image.open(uploaded_file)
        image.verify()
    except (UnidentifiedImageError, OSError):
        st.error("The uploaded file could not be read as a valid image.")
        return

    st.image(image_bytes, caption=uploaded_file.name, use_container_width=True)

    if st.button("Predict mineral", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            try:
                result = request_prediction(image_bytes)
            except requests.ConnectionError:
                st.error(
                    "The prediction service is not running. Start the API locally or set "
                    "MINERAL_API_URL to the deployed service URL."
                )
            except requests.Timeout:
                st.error("The prediction service took too long to respond. Please try again.")
            except requests.HTTPError as error:
                detail = error.response.text if error.response is not None else str(error)
                st.error(f"The prediction service returned an error: {detail}")
            except (requests.RequestException, ValueError) as error:
                st.error(f"The prediction could not be completed: {error}")
            else:
                show_prediction(result)

    with st.expander("Photo tips and limitations"):
        st.markdown(
            """
            - Photograph one mineral specimen at a time.
            - Keep the specimen centered and in focus.
            - Use even lighting and avoid strong reflections.
            - The classifier recognizes only the listed mineral classes.
            - Color and appearance alone cannot replace hardness, streak, cleavage, or expert testing.
            """
        )


if __name__ == "__main__":
    main()
