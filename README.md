# Video Privacy Editor

A sophisticated tool for detecting and blurring sensitive information in screen recordings, with support for motion tracking, temporal consistency, and reviewable blur timelines.

## Overview

This tool automatically detects and blurs sensitive areas in screen recordings, such as:
- ChatGPT conversation history
- Terminal command history (particularly Atuin interface)
- Any other configurable sensitive regions

The system uses a multi-stage pipeline that allows for manual review and adjustment before final processing, ensuring both privacy and video quality.

## Key Features

- **Intelligent Detection**: Template matching with color masking for accurate identification
- **Motion Tracking**: Handles moving windows and macOS animation effects
- **Reviewable Pipeline**: Generate blur timelines that can be reviewed and edited before processing
- **Configurable Quality**: Adjustable frame sampling rates for testing vs. production
- **Extensible Architecture**: Easy to add new detectors and blur filters
- **Batch Processing**: Efficient handling of 2-4 hour videos
- **Confidence Scoring**: Automatic flagging of low-confidence detections for manual review

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/miVideoEditor.git
cd miVideoEditor

# Install dependencies
pip install -r requirements.txt

# Verify FFmpeg installation
ffmpeg -version
```

### Basic Usage

1. **Define sensitive scenes** (manual timestamp marking):
```bash
python annotate.py mark-scenes input_video.mp4 --output scenes.json
```

2. **Annotate training data** (draw bounding boxes):
```bash
python annotate.py extract-frames input_video.mp4 scenes.json --output annotations/
python annotate.py label annotations/ --port 8080  # Opens web UI
```

3. **Train detector** (learn from annotations):
```bash
python train.py annotations/ --detector chatgpt --output models/chatgpt_detector.pkl
```

4. **Generate blur timeline**:
```bash
python detect.py input_video.mp4 --model models/chatgpt_detector.pkl --output timeline.json
```

5. **Review and adjust** (optional):
```bash
python review.py timeline.json --video input_video.mp4 --port 8080
```

6. **Process video**:
```bash
python blur.py input_video.mp4 timeline.json --output output_video.mp4
```

### Quick Processing (One Command)

For videos with known sensitive regions:
```bash
python process.py input_video.mp4 \
    --scenes scenes.json \
    --models models/ \
    --output output_video.mp4 \
    --quality high
```

## Configuration

### Scene Definition Format

```json
{
  "video_path": "recording.mp4",
  "sensitive_scenes": [
    {
      "start": "00:05:30",
      "end": "00:08:45",
      "type": "chatgpt",
      "description": "ChatGPT conversation about project"
    },
    {
      "start": "00:12:00",
      "end": "00:12:30",
      "type": "atuin",
      "description": "Terminal history search"
    }
  ]
}
```

### Processing Modes

- **fast**: Sample every 30 frames (1 second at 30fps) - for testing
- **balanced**: Sample every 10 frames - good quality/speed tradeoff  
- **high**: Sample every 5 frames - high quality
- **maximum**: Process every frame - highest quality, slowest

### Blur Filters

Available blur filters can be configured in the timeline:
- `gaussian`: Standard Gaussian blur
- `pixelate`: Mosaic/pixelation effect
- `noise`: Random noise overlay
- `composite`: Combination of multiple filters (most destructive)

## Architecture

The system follows a modular pipeline architecture:

```
Input Video  Scene Marking  Frame Extraction  Annotation 
 Detector Training  Detection  Timeline Generation 
 Review/Edit  Video Processing  Output Video
```

Each stage produces artifacts that can be reviewed and modified before proceeding.

## Web Interface

The annotation and review tools include a web-based interface for ease of use:

- **Annotation UI**: Draw bounding boxes on extracted frames
- **Review UI**: Visualize detected regions with confidence scores
- **Timeline Editor**: Adjust blur regions and timing

Access the web interface at `http://localhost:8080` when running annotation or review commands.

## Advanced Usage

### Custom Detectors

Add your own detector by extending `BaseDetector`:

```python
from core.detectors import BaseDetector

class MyCustomDetector(BaseDetector):
    def detect(self, frame):
        # Your detection logic
        return DetectionResult(...)
```

### Custom Blur Filters

Create custom blur effects:

```python
from core.filters import BaseBlurFilter

class MyBlurFilter(BaseBlurFilter):
    def apply(self, image, region):
        # Your blur implementation
        return blurred_image
```

### Batch Processing

Process multiple videos:

```bash
python batch_process.py video_list.txt --config batch_config.json
```

## Performance Considerations

- **Memory**: ~4GB RAM recommended for 1080p videos
- **Processing Speed**: 
  - Fast mode: ~10x realtime
  - High mode: ~2x realtime
  - Maximum mode: ~0.5x realtime
- **Storage**: Annotations require ~100MB per hour of video

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Ensure FFmpeg is installed and in PATH
2. **Out of memory**: Reduce frame sampling rate or process in segments
3. **Poor detection**: Add more training annotations or adjust color masks
4. **Blur artifacts**: Use temporal smoothing or adjust blur strength

### Debug Mode

Enable verbose logging:
```bash
python detect.py input.mp4 --debug --log-level DEBUG
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/miVideoEditor/issues)
- Documentation: See [DESIGN.md](DESIGN.md) and [SPECS.md](SPECS.md) for technical details

## Acknowledgments

- FFmpeg for video processing
- OpenCV for computer vision operations
- FastAPI for web interface framework