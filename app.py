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


def classify_image(img: Image.Image, base_url: str) -> str:
    url = base_url.rstrip("/") + "/classify/"
    files = {
        "file": ("image.jpg", _image_to_jpeg_bytes(img), "image/jpeg")
    }

    MAX_RETRIES = 5
    TIMEOUT = 30  # seconds per request
    BACKOFF_FACTOR = 2

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, files=files, timeout=TIMEOUT)
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
        
        base_url = gr.Textbox(value="https://mlops03-latest.onrender.com", label="API base URL")
        
        with gr.Tabs():
            with gr.TabItem("Classify"):
                with gr.Row():
                    with gr.Column():
                        cls_inp = gr.Image(type="pil", label="Upload Image")
                        classify_btn = gr.Button("Classify")
                    with gr.Column():
                        class_out = gr.Textbox(label="Predicted class")
                        cls_status = gr.Textbox(label="Status / Errors")
                
                def _on_classify(image, url):
                    if image is None:
                        return "", "No image provided"
                    res = classify_image(image, url)
                    return res, ""
                
                classify_btn.click(_on_classify, inputs=[cls_inp, base_url], outputs=[class_out, cls_status])

            with gr.TabItem("Normalize"):
                with gr.Row():
                    with gr.Column():
                        norm_inp = gr.Image(type="pil", label="Upload Image")
                        normalize_btn = gr.Button("Normalize")
                    with gr.Column():
                        norm_out = gr.Image(label="Normalized image")
                        norm_status = gr.Textbox(label="Status / Errors")
                
                def _on_normalize(image, url):
                    if image is None:
                        return None, "No image provided"
                    out_img, err = normalize_image_gr(image, url)
                    return out_img, err

                normalize_btn.click(_on_normalize, inputs=[norm_inp, base_url], outputs=[norm_out, norm_status])

            with gr.TabItem("Crop"):
                with gr.Row():
                    with gr.Column():
                        crop_inp = gr.Image(type="pil", label="Upload Image")
                        width = gr.Number(value=64, precision=0, label="Crop width")
                        height = gr.Number(value=64, precision=0, label="Crop height")
                        crop_btn = gr.Button("Crop")
                    with gr.Column():
                        crop_out = gr.Image(label="Cropped image")
                        crop_status = gr.Textbox(label="Status / Errors")

                def _on_crop(image, w, h, url):
                    if image is None:
                        return None, "No image provided"
                    out_img, err = crop_image_gr(image, int(w), int(h), url)
                    return out_img, err

                crop_btn.click(_on_crop, inputs=[crop_inp, width, height, base_url], outputs=[crop_out, crop_status])

    return demo


if __name__ == "__main__":
	demo = build_ui()
	demo.launch(server_name="0.0.0.0", server_port=7860)

