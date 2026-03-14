# Vehicle Counter & Classifier

Real-time vehicle detection, tracking, and counting system built with **YOLOv8**, **Supervision**, and **FastAPI**.

Upload a traffic video → get per-class vehicle counts, an annotated output video, a markdown report, and a bar chart — all through a REST API.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Nano-orange)

---

## Features

- **YOLOv8n** pretrained on COCO (no training required)
- **Multi-class detection**: Car, Motorcycle, Bus, Truck
- **ByteTrack** tracking — unique ID per vehicle across frames
- **Configurable counting line** via API parameters
- **Annotated output video** with bounding boxes, labels, tracker IDs, FPS overlay, and live counter panel
- **Background processing** via subprocess — API stays responsive
- **Reports**: CSV event log, Markdown summary, bar chart PNG
- **Frame skipping** for faster processing on CPU

## Project Structure

```
├── app.py              # FastAPI server — all endpoints
├── pipeline.py         # Core CV pipeline (detect → track → count → annotate → report)
├── worker.py           # Subprocess worker (runs pipeline without blocking the API)
├── config.py           # All configuration (paths, model, thresholds, API settings)
├── schemas.py          # Pydantic request/response models
├── requirements.txt    # Python dependencies
└── README.md
```

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/Vehicle_Counter_Classifier.git
cd Vehicle_Counter_Classifier
pip install -r requirements.txt
```

> YOLOv8n weights are downloaded automatically on first run.

### 2. Run the API

```bash
python app.py
```

Server starts at **http://localhost:8000** — interactive docs at **http://localhost:8000/docs**

### 3. Use It

**Step 1** — Upload a video:
```bash
curl -X POST http://localhost:8000/upload -F "file=@traffic.mp4"
# → {"job_id": "a3f7b2c1d4e5", "filename": "traffic.mp4", "status": "uploaded"}
```

**Step 2** — Start processing:
```bash
curl -X POST http://localhost:8000/process/a3f7b2c1d4e5 \
  -H "Content-Type: application/json" \
  -d '{"line_position": 0.65, "confidence": 0.3}'
# → {"job_id": "a3f7b2c1d4e5", "status": "processing"}
```

**Step 3** — Poll status:
```bash
curl http://localhost:8000/status/a3f7b2c1d4e5
# → {"job_id": "a3f7b2c1d4e5", "status": "processing", "progress_percent": 45.2, "counts": {"Car": 23}}
```

**Step 4** — Get results (once status is `"done"`):
```bash
curl http://localhost:8000/results/a3f7b2c1d4e5
# → {"total_vehicles": 120, "counts": {"Car": 112, "Bus": 4, "Truck": 4, "Motorcycle": 0}}
```

**Step 5** — Download outputs:
```bash
curl -O http://localhost:8000/download/video/a3f7b2c1d4e5    # annotated video
curl -O http://localhost:8000/download/report/a3f7b2c1d4e5   # markdown report
curl -O http://localhost:8000/download/chart/a3f7b2c1d4e5    # bar chart PNG
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — model status and device |
| `POST` | `/upload` | Upload video (mp4/avi, max 100MB) |
| `POST` | `/process/{job_id}` | Start detection pipeline (background) |
| `GET` | `/status/{job_id}` | Poll job progress |
| `GET` | `/results/{job_id}` | Get final counts |
| `GET` | `/download/video/{job_id}` | Download annotated video |
| `GET` | `/download/report/{job_id}` | Download markdown report |
| `GET` | `/download/chart/{job_id}` | Download bar chart PNG |

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_SIZE` | `yolov8n.pt` | YOLO model variant |
| `CONFIDENCE_THRESHOLD` | `0.3` | Minimum detection confidence |
| `LINE_POSITION` | `0.5` | Counting line position (0.0=top, 1.0=bottom) |
| `FRAME_SKIP` | `2` | Process every Nth frame |
| `MAX_FILE_SIZE_MB` | `100` | Maximum upload size |
| `API_HOST` | `0.0.0.0` | Server host |
| `API_PORT` | `8000` | Server port |

## Processing Parameters

When calling `POST /process/{job_id}`, you can pass:

```json
{
  "line_position": 0.65,
  "confidence": 0.3
}
```

- **`line_position`** (0.0–1.0): Where the counting line sits. `0.65` (lower third) is recommended — vehicles are larger and fully visible.
- **`confidence`** (0.05–1.0): Detection threshold. Lower = more detections (may include false positives). Higher = stricter.

## Architecture

```
Client → POST /upload → saves video to data/
       → POST /process → launches worker.py via subprocess
                          worker.py → pipeline.run_pipeline()
                                      YOLOv8 detect → ByteTrack → count → annotate
                                      writes status.json every 30 frames
       → GET /status → reads status.json from disk (instant response)
       → GET /results → reads final status.json
       → GET /download/* → streams output files
```

**Why subprocess?** YOLO inference is CPU-heavy. Python's GIL blocks threads, and `multiprocessing` has Windows compatibility issues. `subprocess.Popen` launches a fully independent process that communicates via a JSON file on disk — simple and reliable.

## Output Files

For each processed job, the pipeline generates:

- `{job_id}_annotated.mp4` — Video with bounding boxes, tracker IDs, counting line, and live stats overlay
- `{job_id}_events.csv` — Every crossing event: `timestamp_sec, vehicle_type, tracker_id`
- `{job_id}_report.md` — Markdown summary with counts table
- `{job_id}_chart.png` — Bar chart of vehicle counts by type

## Tech Stack

- **[YOLOv8](https://docs.ultralytics.com/)** — Object detection (pretrained on COCO)
- **[Supervision](https://supervision.roboflow.com/)** — ByteTrack tracking, annotations, line zones
- **[FastAPI](https://fastapi.tiangolo.com/)** — REST API framework
- **[OpenCV](https://opencv.org/)** — Video I/O and frame manipulation
- **[Matplotlib](https://matplotlib.org/)** — Chart generation

## License

MIT
