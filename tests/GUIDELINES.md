# Test Guidelines for miVideoEditor

## Core Testing Principles

### 1. Test Structure
- Mirror the source code structure in the `tests/` directory
- Each module should have corresponding test files prefixed with `test_`
- Group related tests in classes when appropriate
- Use descriptive test names that explain what is being tested

### 2. Test Coverage Requirements
- **Core modules**: Minimum 95% coverage
- **Utils modules**: Minimum 90% coverage  
- **Storage/Detection/Processing**: Minimum 85% coverage
- **Web module**: Minimum 80% coverage

### 3. Test Types

#### Unit Tests
- Test individual functions/methods in isolation
- Mock external dependencies (file system, network, databases)
- Focus on edge cases and error conditions
- Should be fast (<100ms per test)

#### Integration Tests
- Test interaction between modules
- Use real implementations where possible
- Located in `tests/integration/`
- Can be slower but should complete within seconds

#### Property-Based Tests
- Use Hypothesis for generating test cases
- Focus on invariants and properties that should always hold
- Particularly important for data models and validators

## Best Practices

### 1. Test Organization
```python
import pytest
from hypothesis import given, strategies as st

class TestBoundingBox:
    """Tests for BoundingBox model."""
    
    def test_valid_creation(self):
        """Test creating a valid bounding box."""
        
    def test_invalid_dimensions_raises_error(self):
        """Test that invalid dimensions raise ValidationError."""
        
    @given(st.integers(min_value=1), st.integers(min_value=1))
    def test_area_calculation(self, width: int, height: int):
        """Test area calculation with random valid dimensions."""
```

### 2. Fixtures
```python
@pytest.fixture
def sample_bounding_box():
    """Provide a standard bounding box for testing."""
    return BoundingBox(x=10, y=20, width=100, height=50)

@pytest.fixture
def video_frame():
    """Provide a sample video frame."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)
```

### 3. Assertions
- Use specific assertions with clear messages
- Test both success and failure cases
- Verify exception types and messages
```python
# Good
with pytest.raises(ValidationError, match="width must be positive"):
    BoundingBox(x=0, y=0, width=-1, height=10)

# Avoid
try:
    BoundingBox(x=0, y=0, width=-1, height=10)
    assert False
except:
    pass
```

### 4. Mocking
```python
from unittest.mock import Mock, patch, MagicMock

@patch('mivideoeditor.utils.video.ffmpeg')
def test_video_processing(mock_ffmpeg):
    """Test video processing with mocked FFmpeg."""
    mock_ffmpeg.probe.return_value = {'streams': [...]}
```

### 5. Parametrized Tests
```python
@pytest.mark.parametrize("x,y,width,height,expected_area", [
    (0, 0, 10, 10, 100),
    (5, 5, 20, 30, 600),
    (100, 100, 1, 1, 1),
])
def test_bounding_box_area(x, y, width, height, expected_area):
    """Test area calculation with various inputs."""
    bbox = BoundingBox(x=x, y=y, width=width, height=height)
    assert bbox.area == expected_area
```

### 6. Test Data
- Use realistic test data that represents actual use cases
- Store large test fixtures in separate files
- Create helper functions for generating test data
```python
def create_test_timeline(num_regions: int = 5) -> Timeline:
    """Create a timeline with specified number of blur regions."""
    regions = [
        create_test_blur_region(start_time=i*10, end_time=(i+1)*10)
        for i in range(num_regions)
    ]
    return Timeline(
        video_path=Path("/test/video.mp4"),
        video_duration=100.0,
        frame_rate=30.0,
        blur_regions=regions
    )
```

## Testing Pydantic Models

### 1. Validation Testing
```python
def test_model_validation():
    """Test Pydantic model validation."""
    # Test valid input
    valid_data = {"x": 10, "y": 20, "width": 100, "height": 50}
    bbox = BoundingBox(**valid_data)
    assert bbox.x == 10
    
    # Test invalid input
    invalid_data = {"x": 10, "y": 20, "width": -100, "height": 50}
    with pytest.raises(ValidationError) as exc_info:
        BoundingBox(**invalid_data)
    assert "width must be positive" in str(exc_info.value)
```

### 2. Serialization Testing
```python
def test_model_serialization():
    """Test model serialization/deserialization."""
    bbox = BoundingBox(x=10, y=20, width=100, height=50)
    
    # Test dict export
    data = bbox.model_dump()
    assert data == {"x": 10, "y": 20, "width": 100, "height": 50}
    
    # Test JSON export
    json_str = bbox.model_dump_json()
    
    # Test reconstruction
    bbox2 = BoundingBox.model_validate_json(json_str)
    assert bbox == bbox2
```

## Performance Testing

### 1. Benchmark Critical Paths
```python
import time

def test_detection_performance(benchmark):
    """Test detection algorithm performance."""
    frame = create_test_frame()
    detector = TemplateDetector()
    
    result = benchmark(detector.detect, frame)
    assert result.detection_time < 0.1  # Must complete in 100ms
```

### 2. Memory Usage
```python
import tracemalloc

def test_memory_usage():
    """Test memory consumption stays within limits."""
    tracemalloc.start()
    
    # Perform memory-intensive operation
    processor = VideoProcessor()
    processor.process_large_video()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    assert peak < 1_000_000_000  # Less than 1GB
```

## Error Testing

### 1. Exception Handling
```python
def test_error_recovery():
    """Test system recovers gracefully from errors."""
    processor = VideoProcessor()
    
    # Test with invalid input
    with pytest.raises(ProcessingError) as exc_info:
        processor.process_video(Path("/nonexistent.mp4"))
    
    # Verify error details
    assert exc_info.value.error_code == "FILE_NOT_FOUND"
    assert exc_info.value.recoverable is False
```

### 2. Edge Cases
```python
def test_edge_cases():
    """Test boundary conditions and edge cases."""
    # Test minimum valid values
    bbox = BoundingBox(x=0, y=0, width=1, height=1)
    assert bbox.area == 1
    
    # Test maximum values
    bbox = BoundingBox(x=0, y=0, width=3840, height=2160)  # 4K resolution
    assert bbox.area == 3840 * 2160
```

## Test Execution

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mivideoeditor --cov-report=html

# Run specific test file
pytest tests/core/test_models.py

# Run with verbose output
pytest -v

# Run only marked tests
pytest -m "unit"

# Run with parallel execution
pytest -n auto
```

### Continuous Integration
- All tests must pass before merging
- Coverage reports should be generated
- Performance benchmarks should be tracked
- Test results should be easily accessible

## Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit
def test_unit_functionality():
    """Fast unit test."""

@pytest.mark.integration
def test_integration():
    """Integration test that may be slower."""

@pytest.mark.slow
def test_large_video_processing():
    """Test that takes significant time."""

@pytest.mark.gpu
def test_gpu_acceleration():
    """Test requiring GPU hardware."""
```

## Documentation

Every test should have:
1. Clear, descriptive name
2. Docstring explaining what is being tested
3. Comments for complex logic
4. Clear assertion messages

## Maintenance

- Remove obsolete tests promptly
- Update tests when requirements change
- Refactor tests to reduce duplication
- Keep test execution time reasonable
- Monitor and fix flaky tests immediately