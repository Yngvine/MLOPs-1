import io
import requests
from typing import Tuple, Optional
import time

from PIL import Image

import gradio as gr

def _image_to_jpeg_bytes(img: Image.Image) -> bytes:
	buf = io.BytesIO()
	img.save(buf, format="JPEG")
	return buf.getvalue()


def classify_image(img: Image.Image, n_classes: int, base_url: str) -> str:
    if n_classes is None or n_classes <= 0:
        return "n_classes must be > 0"
    url = base_url.rstrip("/") + "/classify/"
    files = {
        "file": ("image.jpg", _image_to_jpeg_bytes(img), "image/jpeg")
    }
    data = {"n_classes": str(n_classes)}

    MAX_RETRIES = 5
    TIMEOUT = 30  # seconds per request
    BACKOFF_FACTOR = 2

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, files=files, data=data, timeout=TIMEOUT)
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                return f"Request failed after {attempt} attempts: {e}"
            time.sleep(BACKOFF_FACTOR ** (attempt - 1))
            continue
        if resp.status_code != 200:
            if attempt == MAX_RETRIES:
                return f"Error {resp.status_code}: {resp.text}"
            time.sleep(BACKOFF_FACTOR ** (attempt - 1))
            continue
        try:
            j = resp.json()
            return str(j.get("predicted_class", j))
        except Exception:
            return "Invalid JSON response"
    # fallback
    return "Request failed"

def _call_image_return(img: Image.Image, endpoint: str, base_url: str, extra: Optional[dict] = None) -> Tuple[Optional[Image.Image], str]:
    url = base_url.rstrip("/") + endpoint
    files = {"file": ("image.jpg", _image_to_jpeg_bytes(img), "image/jpeg")}
    data = {}
    if extra:
        data.update({k: str(v) for k, v in extra.items()})

    MAX_RETRIES = 6
    TIMEOUT = 40  # seconds per request
    BACKOFF_FACTOR = 2

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, files=files, data=data, timeout=TIMEOUT)
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                return None, f"Request failed after {attempt} attempts: {e}"
            time.sleep(BACKOFF_FACTOR ** (attempt - 1))
            continue
        if resp.status_code != 200:
            if attempt == MAX_RETRIES:
                return None, f"Error {resp.status_code}: {resp.text}"
            time.sleep(BACKOFF_FACTOR ** (attempt - 1))
            continue
        try:
            out_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return out_img, ""
        except Exception as e:
            if attempt == MAX_RETRIES:
                return None, f"Failed to decode image after {attempt} attempts: {e}"
            time.sleep(BACKOFF_FACTOR ** (attempt - 1))
            continue

    # fallback
    return None, "Request failed"


def normalize_image_gr(img: Image.Image, base_url: str) -> Tuple[Optional[Image.Image], str]:
	out_img, err = _call_image_return(img, "/normalize/", base_url)
	if err:
		return None, err
	return out_img, ""


def crop_image_gr(img: Image.Image, width: int, height: int, base_url: str) -> Tuple[Optional[Image.Image], str]:
	if width is None or height is None or width <= 0 or height <= 0:
		return None, "width and height must be positive integers"
	out_img, err = _call_image_return(img, "/crop/", base_url, extra={"width": width, "height": height})
	if err:
		return None, err
	return out_img, ""


def build_ui():
	with gr.Blocks() as demo:
		gr.Markdown("# Gradio frontend for FastAPI image service")
		with gr.Row():
			with gr.Column(scale=1):
				inp = gr.Image(type="pil", label="Upload Image")
				n_classes = gr.Number(value=3, precision=0, label="n_classes (for classify)")
				width = gr.Number(value=64, precision=0, label="Crop width")
				height = gr.Number(value=64, precision=0, label="Crop height")
				base_url = gr.Textbox(value="https://mlops02-latest.onrender.com", label="API base URL")
				classify_btn = gr.Button("Classify")
				normalize_btn = gr.Button("Normalize")
				crop_btn = gr.Button("Crop")
			with gr.Column(scale=1):
				class_out = gr.Textbox(label="Predicted class")
				norm_out = gr.Image(label="Normalized image")
				crop_out = gr.Image(label="Cropped image")
				status = gr.Textbox(label="Status / Errors")

		def _on_classify(image, n, url):
			if image is None:
				return "", None, None, "No image provided"
			res = classify_image(image, int(n), url)
			return res, None, None, ""

		def _on_normalize(image, url):
			if image is None:
				return None, "No image provided"
			out_img, err = normalize_image_gr(image, url)
			return out_img, err

		def _on_crop(image, w, h, url):
			if image is None:
				return None, "No image provided"
			out_img, err = crop_image_gr(image, int(w), int(h), url)
			return out_img, err

		classify_btn.click(_on_classify, inputs=[inp, n_classes, base_url], outputs=[class_out, norm_out, crop_out, status])
		normalize_btn.click(_on_normalize, inputs=[inp, base_url], outputs=[norm_out, status])
		crop_btn.click(_on_crop, inputs=[inp, width, height, base_url], outputs=[crop_out, status])

	return demo


if __name__ == "__main__":
	demo = build_ui()
	demo.launch(server_name="0.0.0.0", server_port=7860)

