# Dataset Manager

Web-service for dataset housekeeping in projects.

## What It Does

- Accepts separate `images` and `labels` folders, or mixed labels in the same images folder.
- Supports optional classes file (`.yaml/.yml/.txt`) for class-name interpretation in analytics and exports.
- Finds pairs and orphans (`image without label`, `label without image`) with checkbox-based deletion flow.
- Checks duplicates inside current dataset and lets you keep one file per duplicate group.
- Moves or copies labels into a dedicated folder.
- Builds `train/val/test` split with custom ratios and optional YOLO `data.yaml`.
- Merges existing datasets and detects duplicate files (by image hash), with duplicate resolution and optional source cleanup.
  - Optional YOLO YAML generation for merged output.
  - Supports split-oriented layouts, including:
    - `images/train`, `labels/train` style
    - `train/images`, `train/labels` style
  - Merged output layout: `train/images`, `train/labels`, `val/images`, `val/labels`, `test/images`, `test/labels`
- Generates dataset charts:
  - class distribution
  - objects per image
  - bbox width/height
  - bbox area
  - pair coverage
  - image size distribution
  - If output folder is empty, charts are generated and shown in UI without saving files to disk.

## Install

```powershell
cd yolo_dataset_manager
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
cd yolo_dataset_manager
uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

Open `http://127.0.0.1:8010`.

## Notes

- Split and merge support `copy` and `move` modes.
- Duplicate detection in merge is hash-based for image files.
