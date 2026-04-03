from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .dataset_ops import (
    IMAGE_EXTENSIONS,
    apply_orphan_actions,
    clean_classes,
    create_split,
    find_dataset_duplicates,
    generate_analytics,
    load_classes,
    merge_datasets,
    merge_preview,
    move_labels_to_separate_folder,
    resolve_dataset_duplicates,
    resolve_path,
    scan_dataset,
)
from .schemas import (
    AnalyticsRequest,
    MergePreviewRequest,
    MergeRunRequest,
    MoveLabelsRequest,
    OrphanActionsRequest,
    PairDuplicatesResolveRequest,
    SessionRequest,
    SplitRequest,
)

@dataclass
class SessionState:
    images_dir: Path | None = None
    labels_dir: Path | None = None
    classes: List[str] = field(default_factory=list)
    classes_file: str | None = None


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


app = FastAPI(title="Dataset Manager", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", NoCacheStaticFiles(directory=str(STATIC_DIR)), name="static")

_state = SessionState()
_lock = Lock()


def _state_snapshot() -> SessionState:
    with _lock:
        return SessionState(
            images_dir=_state.images_dir,
            labels_dir=_state.labels_dir,
            classes=list(_state.classes),
            classes_file=_state.classes_file,
        )


def _require_session() -> SessionState:
    snapshot = _state_snapshot()
    if snapshot.images_dir is None:
        raise HTTPException(status_code=400, detail="Session is not configured. Call POST /api/session first.")
    return snapshot


def _serialize_session(snapshot: SessionState) -> Dict[str, Any]:
    return {
        "images_dir": str(snapshot.images_dir) if snapshot.images_dir else None,
        "labels_dir": str(snapshot.labels_dir) if snapshot.labels_dir else None,
        "classes": snapshot.classes,
        "classes_file": snapshot.classes_file,
    }


def _public_scan(scan: Dict[str, Any]) -> Dict[str, Any]:
    duplicates = find_dataset_duplicates(Path(scan["images_root"]), Path(scan["labels_root"]))
    return {
        "images_root": scan["images_root"],
        "labels_root": scan["labels_root"],
        "image_count": scan["image_count"],
        "label_count": scan["label_count"],
        "pair_count": scan["pair_count"],
        "images_without_labels": scan["images_without_labels"],
        "labels_without_images": scan["labels_without_images"],
        "image_collisions": scan["image_collisions"],
        "label_collisions": scan["label_collisions"],
        "duplicate_group_count": duplicates["duplicate_group_count"],
        "duplicate_item_count": duplicates["duplicate_item_count"],
        "duplicate_groups": duplicates["duplicate_groups"],
        "hash_errors": duplicates["hash_errors"],
    }


def _scan_current(snapshot: SessionState) -> Dict[str, Any]:
    images_dir = snapshot.images_dir
    if images_dir is None:
        raise HTTPException(status_code=400, detail="Session is not configured.")
    try:
        scan = scan_dataset(images_dir, snapshot.labels_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return scan


def _list_roots() -> List[str]:
    if os.name == "nt":
        roots: List[str] = []
        for drive in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            root = Path(f"{drive}:\\")
            if root.exists():
                roots.append(str(root))
        return roots
    return ["/"]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@app.get("/api/session")
def get_session() -> Dict[str, Any]:
    snapshot = _state_snapshot()
    payload = _serialize_session(snapshot)
    if snapshot.images_dir:
        payload["scan"] = _public_scan(_scan_current(snapshot))
    else:
        payload["scan"] = None
    return payload


@app.post("/api/session")
def set_session(payload: SessionRequest) -> Dict[str, Any]:
    images_dir = resolve_path(Path.cwd(), payload.images_dir)
    if not images_dir.exists() or not images_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Images directory does not exist: {images_dir}")

    labels_dir: Path | None = None
    if payload.labels_dir:
        labels_dir = resolve_path(images_dir, payload.labels_dir)
        if not labels_dir.exists() or not labels_dir.is_dir():
            raise HTTPException(status_code=400, detail=f"Labels directory does not exist: {labels_dir}")

    try:
        classes = load_classes(
            images_dir=images_dir,
            provided_classes=[],
            classes_file=payload.classes_file,
            labels_dir=labels_dir,
        )
        scan = scan_dataset(images_dir, labels_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _lock:
        _state.images_dir = images_dir
        _state.labels_dir = labels_dir
        _state.classes = classes
        _state.classes_file = payload.classes_file

    snapshot = _state_snapshot()
    return {"session": _serialize_session(snapshot), "scan": _public_scan(scan)}


@app.post("/api/scan")
def run_scan() -> Dict[str, Any]:
    snapshot = _require_session()
    scan = _scan_current(snapshot)
    return {
        "session": _serialize_session(snapshot),
        "scan": _public_scan(scan),
    }


@app.post("/api/orphans/actions")
def run_orphan_actions(payload: OrphanActionsRequest) -> Dict[str, Any]:
    snapshot = _require_session()
    if snapshot.images_dir is None:
        raise HTTPException(status_code=400, detail="Session is not configured.")

    try:
        result = apply_orphan_actions(
            images_dir=snapshot.images_dir,
            labels_dir=snapshot.labels_dir,
            items=[item.model_dump() for item in payload.items],
        )
        scan = scan_dataset(snapshot.images_dir, snapshot.labels_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"File operation failed: {exc}") from exc

    return {
        "result": result,
        "scan": _public_scan(scan),
    }


@app.post("/api/duplicates/resolve")
def run_duplicates_resolve(payload: PairDuplicatesResolveRequest) -> Dict[str, Any]:
    snapshot = _require_session()
    if snapshot.images_dir is None:
        raise HTTPException(status_code=400, detail="Session is not configured.")
    resolutions = {item.group_id: item.keep_index for item in payload.resolutions}
    try:
        result = resolve_dataset_duplicates(
            images_dir=snapshot.images_dir,
            labels_dir=snapshot.labels_dir,
            resolutions=resolutions,
            delete_skipped_duplicates=True,
        )
        scan = scan_dataset(snapshot.images_dir, snapshot.labels_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Duplicate resolve failed: {exc}") from exc

    return {
        "result": result,
        "scan": _public_scan(scan),
    }


@app.get("/api/image/by-path")
def get_image_by_path(path: str = Query(..., description="Absolute image file path")) -> FileResponse:
    image_path = Path(path).expanduser().resolve()
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail=f"Image file not found: {image_path}")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported image extension: {image_path.suffix}")
    return FileResponse(image_path)


@app.get("/api/text/by-path")
def get_text_by_path(path: str = Query(..., description="Absolute text file path")) -> Dict[str, str]:
    text_path = Path(path).expanduser().resolve()
    if not text_path.exists() or not text_path.is_file():
        raise HTTPException(status_code=404, detail=f"Text file not found: {text_path}")
    if text_path.suffix.lower() != ".txt":
        raise HTTPException(status_code=400, detail=f"Only .txt files are supported: {text_path.suffix}")
    try:
        content = text_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = text_path.read_text(encoding="utf-8", errors="replace")
    return {"path": str(text_path), "content": content}


@app.post("/api/labels/separate")
def separate_labels(payload: MoveLabelsRequest) -> Dict[str, Any]:
    snapshot = _require_session()
    if snapshot.images_dir is None:
        raise HTTPException(status_code=400, detail="Session is not configured.")

    target_dir = None
    if payload.target_labels_dir:
        target_dir = resolve_path(snapshot.images_dir, payload.target_labels_dir)

    try:
        move_result = move_labels_to_separate_folder(
            images_dir=snapshot.images_dir,
            labels_dir=snapshot.labels_dir,
            target_labels_dir=target_dir,
            move_unpaired=payload.move_unpaired,
            mode=payload.mode,
        )
        new_labels_dir = Path(move_result["target_labels_dir"]).resolve()
        scan = scan_dataset(snapshot.images_dir, new_labels_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Move operation failed: {exc}") from exc

    with _lock:
        _state.labels_dir = new_labels_dir

    return {
        "result": move_result,
        "session": _serialize_session(_state_snapshot()),
        "scan": _public_scan(scan),
    }


@app.post("/api/split")
def run_split(payload: SplitRequest) -> Dict[str, Any]:
    snapshot = _require_session()
    if snapshot.images_dir is None:
        raise HTTPException(status_code=400, detail="Session is not configured.")
    output_dir = resolve_path(Path.cwd(), payload.output_dir)

    classes = clean_classes(payload.classes) if payload.classes else list(snapshot.classes)
    try:
        split_result = create_split(
            images_dir=snapshot.images_dir,
            labels_dir=snapshot.labels_dir,
            output_dir=output_dir,
            train_ratio=payload.train_ratio,
            val_ratio=payload.val_ratio,
            test_ratio=payload.test_ratio,
            seed=payload.seed,
            mode=payload.mode,
            include_only_paired=payload.include_only_paired,
            generate_yaml=payload.generate_yaml,
            yaml_name=payload.yaml_name,
            classes=classes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Split failed: {exc}") from exc

    return {"result": split_result}


@app.post("/api/merge/preview")
def run_merge_preview(payload: MergePreviewRequest) -> Dict[str, Any]:
    dataset_dirs = [resolve_path(Path.cwd(), path) for path in payload.dataset_dirs if str(path).strip()]
    try:
        result = merge_preview(dataset_dirs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post("/api/merge/run")
def run_merge(payload: MergeRunRequest) -> Dict[str, Any]:
    dataset_dirs = [resolve_path(Path.cwd(), path) for path in payload.dataset_dirs if str(path).strip()]
    output_dir = resolve_path(Path.cwd(), payload.output_dir)
    resolutions = {item.group_id: item.keep_index for item in payload.resolutions}
    classes = clean_classes(payload.classes) if payload.classes else list(_state_snapshot().classes)

    try:
        result = merge_datasets(
            dataset_dirs=dataset_dirs,
            output_dir=output_dir,
            mode=payload.mode,
            resolutions=resolutions,
            delete_skipped_duplicates=payload.delete_skipped_duplicates,
            generate_yaml=payload.generate_yaml,
            yaml_name=payload.yaml_name,
            classes=classes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Merge failed: {exc}") from exc

    return {"result": result}


@app.post("/api/analytics")
def run_analytics(payload: AnalyticsRequest) -> Dict[str, Any]:
    snapshot = _require_session()
    if snapshot.images_dir is None:
        raise HTTPException(status_code=400, detail="Session is not configured.")

    output_dir = resolve_path(snapshot.images_dir, payload.output_dir) if payload.output_dir else None
    try:
        result = generate_analytics(
            images_dir=snapshot.images_dir,
            labels_dir=snapshot.labels_dir,
            chart_ids=payload.charts,
            output_dir=output_dir,
            classes=snapshot.classes,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Analytics failed: {exc}") from exc
    return result


@app.get("/api/fs/roots")
def get_fs_roots() -> Dict[str, List[str]]:
    return {"roots": _list_roots()}


@app.get("/api/fs/list")
def list_fs(
    path: str | None = Query(default=None, description="Directory path to inspect"),
    mode: str = Query(default="all", description="all|dir|yaml"),
) -> Dict[str, Any]:
    if mode not in {"all", "dir", "yaml"}:
        raise HTTPException(status_code=400, detail="Unsupported mode. Use all, dir or yaml.")

    current = resolve_path(Path.cwd(), path) if path else Path.cwd().resolve()
    if not current.exists() or not current.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {current}")

    try:
        entries = list(current.iterdir())
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Permission denied: {current}") from exc

    directories = sorted([entry for entry in entries if entry.is_dir()], key=lambda item: item.name.lower())
    files = sorted([entry for entry in entries if entry.is_file()], key=lambda item: item.name.lower())

    if mode == "yaml":
        files = [entry for entry in files if entry.suffix.lower() in {".yaml", ".yml", ".txt"}]
    elif mode == "dir":
        files = []

    parent = current.parent
    has_parent = parent != current

    return {
        "current_path": str(current),
        "parent_path": str(parent) if has_parent else None,
        "directories": [{"name": entry.name, "path": str(entry)} for entry in directories],
        "files": [{"name": entry.name, "path": str(entry)} for entry in files],
    }
