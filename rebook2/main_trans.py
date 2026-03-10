import argparse

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="PaddlePaddle/PaddleOCR-VL-1.5",
        help="Model name or local path.",
    )
    parser.add_argument(
        "--image-path",
        default="test_data/img_0.jpg",
        help="Input image path.",
    )
    parser.add_argument(
        "--task",
        choices=sorted(PROMPTS),
        default="ocr",
        help="Recognition task to run.",
    )
    return parser.parse_args()


def load_image(image_path: str, task: str) -> tuple[Image.Image, int]:
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    spotting_upscale_threshold = 1500

    if (
        task == "spotting"
        and orig_w < spotting_upscale_threshold
        and orig_h < spotting_upscale_threshold
    ):
        process_w, process_h = orig_w * 2, orig_h * 2
        image = image.resize((process_w, process_h), Image.Resampling.LANCZOS)

    max_pixels = 2048 * 28 * 28 if task == "spotting" else 1280 * 28 * 28
    return image, max_pixels


def main() -> None:
    args = parse_args()
    image, max_pixels = load_image(args.image_path, args.task)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map=DEVICE
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPTS[args.task]},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": processor.image_processor.min_pixels,
                "longest_edge": max_pixels,
            }
        },
    ).to(model.device)

    print(f"Generating on {DEVICE}...")
    outputs = model.generate(**inputs, max_new_tokens=512)
    print("Decoding...")
    result = processor.decode(outputs[0][inputs["input_ids"].shape[-1] : -1])
    print(result)


if __name__ == "__main__":
    main()
