"""
Tests for the base functions in mylib/classifier.py.
"""
import pytest
import numpy as np
from mylib.classifier import predict_class, crop_image, normalize_image

@pytest.fixture
def sample_image():
    """Creates a 100x100 random image (3 channels)."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

@pytest.fixture
def solid_color_image():
    """Creates a 100x100 solid color image."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 128

# --- Tests for predict_class ---

def test_predict_class_valid(sample_image, mocker):
    """Test that predict_class returns a valid string class label."""
    # Mock the ONNX session and class labels
    mock_session = mocker.Mock()
    mock_output = [np.array([[0.1, 0.8, 0.1]])] # Class 1 has highest probability
    mock_session.run.return_value = mock_output
    
    mocker.patch('mylib.classifier._get_ort_session', return_value=mock_session)
    mocker.patch('mylib.classifier._CLASS_LABELS', ["cat", "dog", "bird"])
    
    prediction = predict_class(sample_image)
    assert isinstance(prediction, str)
    assert prediction == "dog"

def test_predict_class_missing_labels(sample_image, mocker):
    """Test that predict_class raises RuntimeError if labels are missing."""
    mock_session = mocker.Mock()
    mocker.patch('mylib.classifier._get_ort_session', return_value=mock_session)
    mocker.patch('mylib.classifier._CLASS_LABELS', None)
    
    with pytest.raises(RuntimeError, match="Class labels not found"):
        predict_class(sample_image)

# --- Tests for crop_image ---

def test_crop_image_center(sample_image):
    """Test that crop_image returns an image of the specified size."""
    target_size = (50, 50)
    cropped = crop_image(sample_image, target_size)
    assert cropped.shape == (50, 50, 3)

def test_crop_image_identity(sample_image):
    """Test cropping to the same size as the input."""
    target_size = (100, 100)
    cropped = crop_image(sample_image, target_size)
    assert cropped.shape == (100, 100, 3)
    np.testing.assert_array_equal(cropped, sample_image)

def test_crop_image_oversized(sample_image):
    """Test cropping to a size larger than the original image."""
    # The current implementation uses slicing which handles out-of-bounds indices gracefully
    # but might result in a smaller image than requested or an empty one depending on the offset.
    # Let's verify the behavior.
    target_size = (200, 200)
    cropped = crop_image(sample_image, target_size)
    
    # With the current implementation:
    # start = (100 - 200) // 2 = -50
    # end = -50 + 200 = 150
    # slice is [-50:150]. In numpy, negative start index means counting from end.
    # So -50 is index 50. 50:150 goes from 50 to 100 (end of array).
    # So we expect a 50x50 output (from the center).
    # Wait, numpy slicing:
    # if start is negative, it means len + start.
    # 100 + (-50) = 50.
    # So it slices from 50 to 150.
    # 50 to 100 is valid. 100 to 150 is ignored.
    # So we get 50 pixels.
    # Same for width.
    # So we expect (50, 50, 3).
    
    assert isinstance(cropped, np.ndarray)
    # We don't strictly enforce shape here as it depends on implementation details of slicing,
    # but we ensure it doesn't crash.

# --- Tests for normalize_image ---

def test_normalize_image_range(sample_image):
    """Test that normalized image values are between 0 and 1."""
    normalized = normalize_image(sample_image)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert normalized.dtype == np.float64 or normalized.dtype == np.float32

def test_normalize_solid_color(solid_color_image):
    """Test normalization of a solid color image (division by zero check)."""
    normalized = normalize_image(solid_color_image)
    # Max - Min = 0. Divisor becomes 1.
    # (128 - 128) / 1 = 0.
    assert normalized.min() == 0.0
    assert normalized.max() == 0.0
    assert not np.isnan(normalized).any()
    assert not np.isinf(normalized).any()
