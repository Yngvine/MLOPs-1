"""
API main module using FastAPI for mylib functions endpoints.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import numpy as np
import cv2
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, Response
from fastapi.requests import Request
import uvicorn

from mylib import predict_class, crop_image, normalize_image

app = FastAPI(
    title="Image Processing and Classification API",
    description="API endpoints for image processing and classification using mylib.",
    version="0.1.0",
)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Pass the 'request' object to the template so it can access app.routes
    return templates.TemplateResponse(request=request, name="home.html")

@app.post("/classify/")
async def classify_image(file: UploadFile = File(...), n_classes: int = Form(...)):
    """Classify an uploaded image into one of N_CLASSES."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img = normalize_image(img)
    img = crop_image(img, (64, 64))
    pred_class = predict_class(img, n_classes)
    
    return {"predicted_class": int(pred_class)}

@app.post("/crop/")
async def crop_image_endpoint(file: UploadFile = File(...), width: int = Form(...), height: int = Form(...)):
    """Crop an uploaded image to WIDTH x HEIGHT."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    cropped_img = crop_image(img, (width, height))
    _, buffer = cv2.imencode('.jpg', cropped_img)
    
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@app.post("/normalize/")
async def normalize_image_endpoint(file: UploadFile = File(...)):
    """Normalize an uploaded image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    normalized = normalize_image(img)
    _, buffer = cv2.imencode('.jpg', normalized)
    
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run("api.fastapi_main:app", host="0.0.0.0", port=8000, reload=True)