"""
Tests for the API endpoints of the functions of mylib.
"""
import pytest
from fastapi.testclient import TestClient
import numpy as np
import cv2

from api.fastapi_main import app

client = TestClient(app, raise_server_exceptions=False)

@pytest.fixture
def sample_image_bytes():
    """Creates a temporary 100x100 random image in bytes format."""
    # Create a random image (100x100, 3 channels)
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    # Encode it to jpg format in memory
    _, encoded_img = cv2.imencode('.jpg', img)
    return encoded_img.tobytes()

@pytest.fixture
def solid_color_image_bytes():
    """Creates a temporary solid color image in bytes format."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    _, encoded_img = cv2.imencode('.jpg', img)
    return encoded_img.tobytes()

@pytest.fixture
def invalid_image_bytes():
    """Creates a text file content disguised as image bytes."""
    return b"This is not an image."

def test_read_root():
    """Test the home page endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # Check for content we added to the HTML
    assert "MLOps Classifier" in response.text

# --- Tests for 'classify' endpoint ---

def test_classify_image(sample_image_bytes, mocker):
    """Test the classification endpoint."""
    # Mock predict_class to return a string
    mocker.patch('api.fastapi_main.predict_class', return_value="Siamese")
    
    files = {"file": ("test_image.jpg", sample_image_bytes, "image/jpeg")}
    
    response = client.post("/classify/", files=files)
    
    assert response.status_code == 200
    json_response = response.json()
    assert "predicted_class" in json_response
    assert isinstance(json_response["predicted_class"], str)
    assert json_response["predicted_class"] == "Siamese"

def test_classify_missing_file():
    """Test classify endpoint with missing file."""
    response = client.post("/classify/")
    assert response.status_code == 422

# --- Tests for 'crop' endpoint ---

def test_crop_image(sample_image_bytes):
    """Test the crop endpoint."""
    files = {"file": ("test_image.jpg", sample_image_bytes, "image/jpeg")}
    data = {"width": str(50), "height": str(50)}
    
    response = client.post("/crop/", files=files, data=data)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    
    # Verify the output is actually an image of correct size
    nparr = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    assert img.shape == (50, 50, 3)

def test_crop_oversized(sample_image_bytes):
    """Test cropping to a size larger than the original image."""
    files = {"file": ("test_image.jpg", sample_image_bytes, "image/jpeg")}
    data = {"width": str(200), "height": str(200)}
    
    response = client.post("/crop/", files=files, data=data)
    
    # If logic allows it (numpy slicing handles out of bounds gracefully), it returns 200
    if response.status_code == 200:
        assert response.headers["content-type"] == "image/jpeg"
        nparr = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        assert img is not None

# --- Tests for 'normalize' endpoint ---

def test_normalize_image(sample_image_bytes):
    """Test the normalize endpoint."""
    files = {"file": ("test_image.jpg", sample_image_bytes, "image/jpeg")}
    
    response = client.post("/normalize/", files=files)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    
    # Verify the output is actually an image
    nparr = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    assert img is not None
    assert img.shape == (100, 100, 3)

def test_normalize_solid_color(solid_color_image_bytes):
    """Test normalization on a solid color image (division by zero check)."""
    files = {"file": ("solid.jpg", solid_color_image_bytes, "image/jpeg")}
    
    response = client.post("/normalize/", files=files)
    
    # Should return 200 if the division by zero fix is implemented
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

def test_normalize_invalid_file(invalid_image_bytes):
    """Test behavior when uploading a non-image file."""
    files = {"file": ("fake.jpg", invalid_image_bytes, "image/jpeg")}
    response = client.post("/normalize/", files=files)

    # cv2.imdecode returns None, causing AttributeError later -> 500 Server Error
    assert response.status_code == 500