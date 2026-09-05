"""Smallest working diagnostic assistant: upload a chest X-ray, get a label + Grad-CAM."""

from importlib import import_module
from pathlib import Path

from PIL import Image

from src.predict import predict_image

gr = import_module("gradio")

CHECKPOINT = Path("checkpoints/resnet50_binary.pt")
DISCLAIMER = (
    "Research demo only — not a medical device. Do not use for diagnosis. "
    "Without a fine-tuned checkpoint, predictions come from ImageNet-initialized ResNet50 "
    "and will not be clinically meaningful."
)


def diagnose(image: Image.Image | None):
    if image is None:
        raise gr.Error("Upload a chest X-ray image first.")
    ckpt = CHECKPOINT if CHECKPOINT.exists() else None
    result = predict_image(image, checkpoint=ckpt)
    lines = [
        f"**Prediction:** {result['label']} ({result['confidence']:.1%})",
        "",
        f"- normal: {result['probabilities']['normal']:.1%}",
        f"- abnormal: {result['probabilities']['abnormal']:.1%}",
        "",
        DISCLAIMER,
    ]
    return "\n".join(lines), result["overlay"]


def main() -> None:
    demo = gr.Interface(
        fn=diagnose,
        inputs=gr.Image(type="pil", label="Chest X-ray"),
        outputs=[
            gr.Markdown(label="Result"),
            gr.Image(type="pil", label="Grad-CAM overlay"),
        ],
        title="Chest X-ray diagnostic assistant (MVP)",
        description=(
            "Binary ResNet50 (normal vs abnormal) with Grad-CAM. "
            "Fine-tune with `python -m src.train` then restart this app."
        ),
        flagging_mode="never",
    )
    demo.launch()


if __name__ == "__main__":
    main()
