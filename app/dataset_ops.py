from __future__ import annotations

import base64
import hashlib
import io
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import yaml
from PIL import Image, UnidentifiedImageError

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = ("train", "val", "test")
NON_LABEL_TEXT_FILES = {"classes.txt", "labels.txt", "readme.txt"}


@dataclass(frozen=True)
class PairRecord:
    key: str
    image_rel: str
    label_rel: str
    image_abs: Path
    label_abs: Path


@dataclass(frozen=True)
class MergeItem:
    source_dataset: Path
    split: str
    inner_key: str
    image_abs: Path
    label_abs: Path
    image_ext: str
    image_hash: str
    image_size: int


def resolve_path(base: Path, maybe_relative: str) -> Path:
    candidate = Path(maybe_relative).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def ensure_in_base(base: Path, candidate: Path) -> Path:
    base_resolved = base.resolve()
    try:
        candidate.resolve().relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError(f"Path escapes base directory: {candidate}") from exc
    return candidate


def clean_classes(raw_classes: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in raw_classes:
        name = str(item).strip()
        if name:
            out.append(name)
    return out


def load_classes(
    images_dir: Path,
    provided_classes: Iterable[str],
    classes_file: str | None = None,
    labels_dir: Path | None = None,
) -> List[str]:
    classes = clean_classes(provided_classes)
    if classes:
        return classes

    if classes_file:
        classes_path = resolve_path(images_dir, classes_file)
        loaded = _load_classes_file(classes_path)
        if loaded:
            return loaded

    candidates: List[Path] = []
    candidates.extend(images_dir / name for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"))
    if labels_dir and labels_dir != images_dir:
        candidates.extend(labels_dir / name for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"))
    parent = images_dir.parent
    candidates.extend(parent / name for name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"))
    candidates.extend(images_dir / name for name in ("classes.txt",))

    for candidate in candidates:
        if not candidate.exists():
            continue
        loaded = _load_classes_file(candidate)
        if loaded:
            return loaded
    return []


def _load_classes_file(path: Path) -> List[str]:
    if not path.exists():
        return []

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        names = data.get("names")
        if isinstance(names, list):
            return clean_classes(names)
        if isinstance(names, dict):
            indexed = []
            for key, value in names.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                indexed.append((idx, str(value)))
            indexed.sort(key=lambda item: item[0])
            return clean_classes(name for _, name in indexed)
        return []
    return clean_classes(path.read_text(encoding="utf-8").splitlines())


def _to_key(rel_path: str) -> str:
    parts = list(Path(rel_path).with_suffix("").parts)
    if not parts:
        return Path(rel_path).with_suffix("").as_posix()

    split: str | None = None
    if parts and parts[0].lower() in SPLIT_NAMES:
        split = parts.pop(0).lower()

    while parts and parts[0].lower() in {"images", "labels"}:
        parts.pop(0)

    if split is None and parts and parts[0].lower() in SPLIT_NAMES:
        split = parts.pop(0).lower()
        while parts and parts[0].lower() in {"images", "labels"}:
            parts.pop(0)

    if not parts:
        parts = [Path(rel_path).stem]

    key_parts = [split] if split else []
    key_parts.extend(parts)
    return Path(*key_parts).as_posix()


def _relative_paths_by_ext(root: Path, suffixes: set[str]) -> List[str]:
    result: List[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        result.append(path.relative_to(root).as_posix())
    result.sort()
    return result


def _is_likely_label_file(path: Path) -> bool:
    if path.suffix.lower() != ".txt":
        return False
    if path.name.lower() in NON_LABEL_TEXT_FILES:
        return False
    return True


def _is_effectively_empty_label(path: Path) -> bool:
    try:
        if path.stat().st_size == 0:
            return True
    except OSError:
        return False

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="utf-8", errors="replace")
    return not content.strip()


def _collect_images(images_dir: Path) -> tuple[Dict[str, str], List[Dict[str, Any]]]:
    by_key: Dict[str, str] = {}
    collisions: List[Dict[str, Any]] = []
    for rel_path in _relative_paths_by_ext(images_dir, IMAGE_EXTENSIONS):
        key = _to_key(rel_path)
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = rel_path
            continue
        collisions.append({"key": key, "kept": previous, "ignored": rel_path})
    return by_key, collisions


def _collect_labels(labels_root: Path) -> tuple[Dict[str, str], List[Dict[str, Any]]]:
    by_key: Dict[str, str] = {}
    collisions: List[Dict[str, Any]] = []
    for path in sorted(labels_root.rglob("*.txt"), key=lambda item: item.as_posix().lower()):
        if not path.is_file() or not _is_likely_label_file(path):
            continue
        rel_path = path.relative_to(labels_root).as_posix()
        key = _to_key(rel_path)
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = rel_path
            continue
        collisions.append({"key": key, "kept": previous, "ignored": rel_path})
    return by_key, collisions


def scan_dataset(images_dir: Path, labels_dir: Path | None = None) -> Dict[str, Any]:
    images_root = images_dir.resolve()
    labels_root = (labels_dir or images_dir).resolve()
    if not images_root.exists() or not images_root.is_dir():
        raise ValueError(f"Images directory does not exist: {images_root}")
    if not labels_root.exists() or not labels_root.is_dir():
        raise ValueError(f"Labels directory does not exist: {labels_root}")

    image_map, image_collisions = _collect_images(images_root)
    label_map, label_collisions = _collect_labels(labels_root)

    image_keys = set(image_map.keys())
    label_keys_all = set(label_map.keys())

    nonempty_label_map: Dict[str, str] = {}
    for key, rel_path in label_map.items():
        abs_path = ensure_in_base(labels_root, (labels_root / rel_path).resolve())
        if not abs_path.exists() or not abs_path.is_file():
            continue
        if _is_effectively_empty_label(abs_path):
            continue
        nonempty_label_map[key] = rel_path

    label_keys_for_pairing = set(nonempty_label_map.keys())
    paired_keys = sorted(image_keys & label_keys_for_pairing)
    image_orphan_keys = sorted(image_keys - label_keys_for_pairing)
    label_orphan_keys = sorted(label_keys_all - image_keys)

    pairs: List[PairRecord] = []
    for key in paired_keys:
        image_rel = image_map[key]
        label_rel = nonempty_label_map[key]
        pairs.append(
            PairRecord(
                key=key,
                image_rel=image_rel,
                label_rel=label_rel,
                image_abs=(images_root / image_rel).resolve(),
                label_abs=(labels_root / label_rel).resolve(),
            )
        )

    return {
        "images_root": str(images_root),
        "labels_root": str(labels_root),
        "pair_count": len(pairs),
        "image_count": len(image_map),
        "label_count": len(label_map),
        "paired": [{"key": item.key, "image_rel": item.image_rel, "label_rel": item.label_rel} for item in pairs],
        "images_without_labels": [image_map[key] for key in image_orphan_keys],
        "labels_without_images": [label_map[key] for key in label_orphan_keys],
        "image_key_map": image_map,
        "label_key_map": label_map,
        "image_collisions": image_collisions,
        "label_collisions": label_collisions,
    }


def find_dataset_duplicates(images_dir: Path, labels_dir: Path | None = None) -> Dict[str, Any]:
    images_root = images_dir.resolve()
    labels_root = (labels_dir or images_dir).resolve()
    scan = scan_dataset(images_root, labels_root)
    image_key_map: Dict[str, str] = dict(scan["image_key_map"])
    label_key_map: Dict[str, str] = dict(scan["label_key_map"])

    by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    hash_errors: List[str] = []

    for key, image_rel in image_key_map.items():
        image_abs = ensure_in_base(images_root, (images_root / image_rel).resolve())
        if not image_abs.exists():
            continue
        try:
            image_hash = file_sha1(image_abs)
            image_size = image_abs.stat().st_size
        except OSError as exc:
            hash_errors.append(f"{image_abs}: {exc}")
            continue

        label_rel = label_key_map.get(key)
        label_abs: str | None = None
        if label_rel:
            resolved_label = ensure_in_base(labels_root, (labels_root / label_rel).resolve())
            label_abs = str(resolved_label) if resolved_label.exists() else None

        by_hash[image_hash].append(
            {
                "key": key,
                "image_rel": image_rel,
                "label_rel": label_rel,
                "image_path": str(image_abs),
                "label_path": label_abs,
                "size_bytes": image_size,
            }
        )

    duplicate_groups: List[Dict[str, Any]] = []
    for hash_key, entries in sorted(by_hash.items(), key=lambda item: item[0]):
        if len(entries) < 2:
            continue
        duplicate_groups.append(
            {
                "group_id": hash_key,
                "count": len(entries),
                "entries": entries,
            }
        )

    return {
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_item_count": sum(group["count"] for group in duplicate_groups),
        "duplicate_groups": duplicate_groups,
        "hash_errors": hash_errors,
    }


def resolve_dataset_duplicates(
    images_dir: Path,
    labels_dir: Path | None,
    resolutions: Dict[str, int],
    delete_skipped_duplicates: bool = True,
) -> Dict[str, Any]:
    duplicates = find_dataset_duplicates(images_dir, labels_dir)
    skipped_entries: List[Dict[str, Any]] = []
    kept_entries: List[Dict[str, Any]] = []

    for group in duplicates["duplicate_groups"]:
        entries = list(group["entries"])
        keep_index = int(resolutions.get(group["group_id"], 0))
        if keep_index < 0 or keep_index >= len(entries):
            keep_index = 0
        for index, entry in enumerate(entries):
            if index == keep_index:
                kept_entries.append(entry)
            else:
                skipped_entries.append(entry)

    deleted_files_count = 0
    delete_errors: List[str] = []
    deleted_preview: List[str] = []

    if delete_skipped_duplicates:
        for entry in skipped_entries:
            targets = [entry.get("image_path"), entry.get("label_path")]
            for target in targets:
                if not target:
                    continue
                path = Path(target)
                try:
                    if path.exists():
                        path.unlink()
                        deleted_files_count += 1
                        deleted_preview.append(str(path))
                except OSError as exc:
                    delete_errors.append(f"{path}: {exc}")

    return {
        "group_count": duplicates["duplicate_group_count"],
        "duplicate_item_count": duplicates["duplicate_item_count"],
        "kept_count": len(kept_entries),
        "skipped_count": len(skipped_entries),
        "deleted_files_count": deleted_files_count,
        "delete_errors": delete_errors,
        "deleted_preview": deleted_preview[:200],
    }


def apply_orphan_actions(
    images_dir: Path,
    labels_dir: Path | None,
    items: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    images_root = images_dir.resolve()
    labels_root = (labels_dir or images_dir).resolve()
    deleted = 0
    kept = 0
    errors: List[Dict[str, str]] = []

    for item in items:
        kind = str(item.get("kind", "")).strip()
        rel_path = str(item.get("rel_path", "")).strip()
        action = str(item.get("action", "")).strip()
        if not rel_path or kind not in {"image", "label"} or action not in {"keep", "delete"}:
            errors.append({"item": str(item), "error": "Invalid orphan action payload."})
            continue

        root = images_root if kind == "image" else labels_root
        abs_path = ensure_in_base(root, (root / rel_path).resolve())

        if action == "keep":
            kept += 1
            continue

        if not abs_path.exists():
            errors.append({"path": str(abs_path), "error": "File does not exist."})
            continue
        abs_path.unlink()
        deleted += 1

    return {
        "deleted_count": deleted,
        "kept_count": kept,
        "errors": errors,
    }


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_dup{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def move_labels_to_separate_folder(
    images_dir: Path,
    labels_dir: Path | None,
    target_labels_dir: Path | None = None,
    move_unpaired: bool = False,
    mode: str = "move",
) -> Dict[str, Any]:
    images_root = images_dir.resolve()
    labels_root = (labels_dir or images_dir).resolve()
    if mode not in {"move", "copy"}:
        raise ValueError("Unsupported mode for labels transfer. Use move or copy.")
    if target_labels_dir is None:
        if images_root.name.lower() == "images":
            target_labels_dir = images_root.parent / "labels"
        else:
            target_labels_dir = images_root / "labels"
    target_root = target_labels_dir.resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    scan = scan_dataset(images_root, labels_root)
    label_key_map: Dict[str, str] = dict(scan["label_key_map"])
    paired_keys = {item["key"] for item in scan["paired"]}
    keys_to_move = list(label_key_map.keys()) if move_unpaired else sorted(paired_keys)

    moved: List[Dict[str, str]] = []
    skipped = 0
    for key in keys_to_move:
        rel_path = label_key_map.get(key)
        if not rel_path:
            continue
        src = ensure_in_base(labels_root, (labels_root / rel_path).resolve())
        if not src.exists() or not src.is_file():
            skipped += 1
            continue
        destination = (target_root / rel_path).resolve()
        ensure_in_base(target_root, destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination == src:
            skipped += 1
            continue
        destination = _unique_destination(destination)
        if mode == "move":
            shutil.move(str(src), str(destination))
        else:
            shutil.copy2(str(src), str(destination))
        moved.append({"from": str(src), "to": str(destination)})

    return {
        "mode": mode,
        "target_labels_dir": str(target_root),
        "moved_count": len(moved),
        "skipped_count": skipped,
        "moved_preview": moved[:200],
    }


def _copy_or_move(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(src), str(dst))
        return
    shutil.copy2(src, dst)


def _safe_yaml_name(name: str) -> str:
    trimmed = str(name).strip()
    if not trimmed:
        return "data.yaml"
    if not trimmed.lower().endswith((".yaml", ".yml")):
        return f"{trimmed}.yaml"
    return trimmed


def _split_output_key(key: str) -> str:
    path = Path(key)
    parts = list(path.parts)
    if parts and parts[0].lower() in SPLIT_NAMES:
        parts = parts[1:]
    if not parts:
        return path.name or "item"
    return Path(*parts).as_posix()


def create_split(
    images_dir: Path,
    labels_dir: Path | None,
    output_dir: Path,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    mode: str,
    include_only_paired: bool,
    generate_yaml: bool,
    yaml_name: str,
    classes: Iterable[str],
) -> Dict[str, Any]:
    images_root = images_dir.resolve()
    labels_root = (labels_dir or images_dir).resolve()
    out_root = output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    scan = scan_dataset(images_root, labels_root)
    image_map: Dict[str, str] = dict(scan["image_key_map"])
    label_map: Dict[str, str] = dict(scan["label_key_map"])
    paired_keys = [item["key"] for item in scan["paired"]]

    records: List[Dict[str, str | None]] = []
    if include_only_paired:
        for key in paired_keys:
            records.append({"key": key, "image_rel": image_map[key], "label_rel": label_map[key]})
    else:
        for key, image_rel in image_map.items():
            records.append({"key": key, "image_rel": image_rel, "label_rel": label_map.get(key)})

    if not records:
        raise ValueError("Nothing to split. Dataset has no images for the selected criteria.")

    ratio_sum = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if ratio_sum <= 0:
        raise ValueError("At least one split ratio must be greater than zero.")

    normalized = {
        "train": float(train_ratio) / ratio_sum,
        "val": float(val_ratio) / ratio_sum,
        "test": float(test_ratio) / ratio_sum,
    }

    rng = random.Random(seed)
    rng.shuffle(records)
    total = len(records)
    n_train = int(total * normalized["train"])
    n_val = int(total * normalized["val"])
    if n_train + n_val > total:
        n_val = max(0, total - n_train)
    n_test = total - n_train - n_val

    split_boundaries = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, total),
    }

    counts = {"train": 0, "val": 0, "test": 0}
    empty_labels = 0

    for split_name, (start, end) in split_boundaries.items():
        for record in records[start:end]:
            key = str(record["key"])
            output_key = _split_output_key(key)
            image_rel = str(record["image_rel"])
            label_rel = record["label_rel"]
            image_src = ensure_in_base(images_root, (images_root / image_rel).resolve())
            ext = Path(image_rel).suffix.lower()
            image_dst = out_root / "images" / split_name / Path(output_key).with_suffix(ext)
            label_dst = out_root / "labels" / split_name / Path(output_key).with_suffix(".txt")

            _copy_or_move(image_src, image_dst, mode)
            if label_rel:
                label_src = ensure_in_base(labels_root, (labels_root / str(label_rel)).resolve())
                _copy_or_move(label_src, label_dst, mode)
            else:
                label_dst.parent.mkdir(parents=True, exist_ok=True)
                label_dst.write_text("", encoding="utf-8")
                empty_labels += 1

            counts[split_name] += 1

    yaml_path: str | None = None
    class_list = clean_classes(classes)
    if generate_yaml:
        safe_yaml_name = _safe_yaml_name(yaml_name)
        yaml_data: Dict[str, Any] = {
            "path": str(out_root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": class_list,
        }
        if class_list:
            yaml_data["nc"] = len(class_list)
        yaml_target = out_root / safe_yaml_name
        yaml_target.write_text(yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        yaml_path = str(yaml_target)

    return {
        "output_dir": str(out_root),
        "mode": mode,
        "total_items": total,
        "split_counts": counts,
        "empty_labels_created": empty_labels,
        "yaml_path": yaml_path,
    }


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _detect_split_and_inner(key: str) -> tuple[str, str]:
    rel_path = Path(key)
    if rel_path.parts and rel_path.parts[0].lower() in SPLIT_NAMES:
        split = rel_path.parts[0].lower()
        if len(rel_path.parts) == 1:
            inner = rel_path.name
        else:
            inner = Path(*rel_path.parts[1:]).as_posix()
        # `key` is already extension-less; keep dots in stem (e.g. ".jpg.rf.<hash>").
        return split, Path(inner).as_posix()
    return "train", rel_path.as_posix()


def _has_split_layout(root: Path) -> bool:
    for split_name in SPLIT_NAMES:
        split_dir = root / split_name
        if not split_dir.exists() or not split_dir.is_dir():
            continue
        if (split_dir / "images").exists() or (split_dir / "labels").exists():
            return True
    return False


def discover_merge_items(dataset_dir: Path) -> Dict[str, Any]:
    root = dataset_dir.resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")

    if _has_split_layout(root):
        images_dir = root
        labels_dir: Path | None = root
    else:
        images_dir = root / "images" if (root / "images").exists() else root
        labels_dir = root / "labels" if (root / "labels").exists() else None
    scan = scan_dataset(images_dir, labels_dir)
    pairs = scan["paired"]
    labels_root = Path(scan["labels_root"])
    image_root = Path(scan["images_root"])

    items: List[MergeItem] = []
    missing_files = 0
    for pair in pairs:
        key = str(pair["key"])
        image_rel = str(pair["image_rel"])
        label_rel = str(pair["label_rel"])
        split, inner_key = _detect_split_and_inner(key)
        image_abs = ensure_in_base(image_root, (image_root / image_rel).resolve())
        label_abs = ensure_in_base(labels_root, (labels_root / label_rel).resolve())
        if not image_abs.exists() or not label_abs.exists():
            missing_files += 1
            continue
        image_hash = file_sha1(image_abs)
        image_size = image_abs.stat().st_size
        items.append(
            MergeItem(
                source_dataset=root,
                split=split,
                inner_key=inner_key,
                image_abs=image_abs,
                label_abs=label_abs,
                image_ext=image_abs.suffix.lower(),
                image_hash=image_hash,
                image_size=image_size,
            )
        )

    return {
        "dataset_dir": str(root),
        "items": items,
        "pair_count": len(pairs),
        "missing_files": missing_files,
        "images_without_labels": scan["images_without_labels"],
        "labels_without_images": scan["labels_without_images"],
    }


def merge_preview(dataset_dirs: Sequence[Path]) -> Dict[str, Any]:
    if not dataset_dirs:
        raise ValueError("Add at least one dataset directory for merge preview.")

    all_items: List[MergeItem] = []
    sources: List[Dict[str, Any]] = []
    for dataset_dir in dataset_dirs:
        discovered = discover_merge_items(dataset_dir)
        items: List[MergeItem] = discovered["items"]
        all_items.extend(items)
        sources.append(
            {
                "dataset_dir": discovered["dataset_dir"],
                "pair_count": discovered["pair_count"],
                "valid_items": len(items),
                "missing_files": discovered["missing_files"],
                "images_without_labels": len(discovered["images_without_labels"]),
                "labels_without_images": len(discovered["labels_without_images"]),
            }
        )

    by_hash: Dict[str, List[MergeItem]] = defaultdict(list)
    for item in all_items:
        by_hash[item.image_hash].append(item)

    duplicate_groups: List[Dict[str, Any]] = []
    for hash_key, items in sorted(by_hash.items(), key=lambda kv: kv[0]):
        if len(items) < 2:
            continue
        duplicate_groups.append(
            {
                "group_id": hash_key,
                "count": len(items),
                "entries": [
                    {
                        "dataset_dir": str(item.source_dataset),
                        "split": item.split,
                        "inner_key": item.inner_key,
                        "image_path": str(item.image_abs),
                        "label_path": str(item.label_abs),
                        "size_bytes": item.image_size,
                    }
                    for item in items
                ],
            }
        )

    return {
        "total_items": len(all_items),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_item_count": sum(group["count"] for group in duplicate_groups),
        "sources": sources,
        "duplicate_groups": duplicate_groups,
    }


def _destination_with_collision_handling(
    image_dst_base: Path,
    label_dst_base: Path,
    source_hash: str,
) -> tuple[Path, Path, bool]:
    candidate_image = image_dst_base
    candidate_label = label_dst_base
    if candidate_image.exists() and file_sha1(candidate_image) == source_hash:
        return candidate_image, candidate_label, True

    index = 1
    while candidate_image.exists() or candidate_label.exists():
        candidate_image = image_dst_base.with_name(f"{image_dst_base.stem}_dup{index}{image_dst_base.suffix}")
        candidate_label = label_dst_base.with_name(f"{label_dst_base.stem}_dup{index}{label_dst_base.suffix}")
        if candidate_image.exists() and file_sha1(candidate_image) == source_hash:
            return candidate_image, candidate_label, True
        index += 1
    return candidate_image, candidate_label, False


def merge_datasets(
    dataset_dirs: Sequence[Path],
    output_dir: Path,
    *,
    mode: str,
    resolutions: Dict[str, int],
    delete_skipped_duplicates: bool,
    generate_yaml: bool = False,
    yaml_name: str = "data.yaml",
    classes: Iterable[str] = (),
) -> Dict[str, Any]:
    if not dataset_dirs:
        raise ValueError("Add at least one dataset directory to merge.")

    all_items: List[MergeItem] = []
    source_summaries: List[Dict[str, Any]] = []
    for dataset_dir in dataset_dirs:
        discovered = discover_merge_items(dataset_dir)
        all_items.extend(discovered["items"])
        source_summaries.append(
            {
                "dataset_dir": str(discovered["dataset_dir"]),
                "pair_count": int(discovered["pair_count"]),
                "valid_items": len(discovered["items"]),
                "missing_files": int(discovered["missing_files"]),
                "images_without_labels": len(discovered["images_without_labels"]),
                "labels_without_images": len(discovered["labels_without_images"]),
            }
        )

    if not all_items:
        details = "; ".join(
            (
                f"{item['dataset_dir']} "
                f"(pairs={item['pair_count']}, valid={item['valid_items']}, "
                f"img_wo_lbl={item['images_without_labels']}, lbl_wo_img={item['labels_without_images']}, "
                f"missing={item['missing_files']})"
            )
            for item in source_summaries
        )
        raise ValueError(f"No paired image/label files were found for merge. Sources: {details}")

    by_hash: Dict[str, List[MergeItem]] = defaultdict(list)
    for item in all_items:
        by_hash[item.image_hash].append(item)

    selected: List[MergeItem] = []
    skipped_duplicates: List[MergeItem] = []
    for hash_key, items in by_hash.items():
        if len(items) == 1:
            selected.append(items[0])
            continue
        keep_index = int(resolutions.get(hash_key, 0))
        if keep_index < 0 or keep_index >= len(items):
            keep_index = 0
        for index, item in enumerate(items):
            if index == keep_index:
                selected.append(item)
            else:
                skipped_duplicates.append(item)

    deleted_from_sources = 0
    delete_errors: List[str] = []
    if delete_skipped_duplicates:
        for item in skipped_duplicates:
            for target in (item.image_abs, item.label_abs):
                try:
                    if target.exists():
                        target.unlink()
                        deleted_from_sources += 1
                except OSError as exc:
                    delete_errors.append(f"{target}: {exc}")

    out_root = output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    merged_count = 0
    skipped_existing = 0
    renamed_due_collision = 0
    split_counts = {"train": 0, "val": 0, "test": 0}
    renamed_files: List[Dict[str, str]] = []

    for item in selected:
        image_dst_base = out_root / item.split / "images" / Path(item.inner_key).with_suffix(item.image_ext)
        label_dst_base = out_root / item.split / "labels" / Path(item.inner_key).with_suffix(".txt")
        image_dst, label_dst, identical_exists = _destination_with_collision_handling(
            image_dst_base=image_dst_base,
            label_dst_base=label_dst_base,
            source_hash=item.image_hash,
        )

        if identical_exists and image_dst.exists():
            skipped_existing += 1
            if not label_dst.exists():
                _copy_or_move(item.label_abs, label_dst, mode)
            continue

        if image_dst != image_dst_base:
            renamed_due_collision += 1
            renamed_files.append({"from": str(image_dst_base), "to": str(image_dst)})

        _copy_or_move(item.image_abs, image_dst, mode)
        _copy_or_move(item.label_abs, label_dst, mode)
        merged_count += 1
        split_counts[item.split] += 1

    yaml_path: str | None = None
    class_list = clean_classes(classes)
    if generate_yaml:
        safe_yaml_name = _safe_yaml_name(yaml_name)
        yaml_data: Dict[str, Any] = {
            "path": str(out_root),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": class_list,
        }
        if class_list:
            yaml_data["nc"] = len(class_list)
        yaml_target = out_root / safe_yaml_name
        yaml_target.write_text(yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        yaml_path = str(yaml_target)

    return {
        "output_dir": str(out_root),
        "total_candidates": len(all_items),
        "selected_after_duplicate_filter": len(selected),
        "duplicate_skipped_count": len(skipped_duplicates),
        "merged_count": merged_count,
        "skipped_existing_count": skipped_existing,
        "renamed_due_collision": renamed_due_collision,
        "split_counts": split_counts,
        "deleted_from_sources_count": deleted_from_sources,
        "delete_errors": delete_errors,
        "renamed_preview": renamed_files[:200],
        "yaml_path": yaml_path,
    }


def parse_yolo_label_file(path: Path) -> List[Dict[str, float | int]]:
    records: List[Dict[str, float | int]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue
        records.append(
            {
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "area": width * height,
            }
        )
    return records


def _fig_to_file(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _fig_to_base64(fig: Any) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_to_base64(path: Path) -> str:
    raw = path.read_bytes()
    return base64.b64encode(raw).decode("ascii")


def _chart_output_path(output_dir: Path, chart_id: str) -> Path:
    return output_dir / f"{chart_id}.png"


def _safe_ratio(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return part / total


def _build_chart_payload(fig: Any, chart_id: str, title: str, output_root: Path | None) -> Dict[str, Any]:
    if output_root is None:
        return {
            "id": chart_id,
            "title": title,
            "file_path": None,
            "image_base64": _fig_to_base64(fig),
        }

    output_path = _chart_output_path(output_root, chart_id)
    _fig_to_file(fig, output_path)
    return {
        "id": chart_id,
        "title": title,
        "file_path": str(output_path),
        "image_base64": _image_to_base64(output_path),
    }


def generate_analytics(
    images_dir: Path,
    labels_dir: Path | None,
    chart_ids: Sequence[str],
    output_dir: Path | None,
    classes: Iterable[str],
) -> Dict[str, Any]:
    if plt is None:
        raise RuntimeError("matplotlib is not installed. Install requirements from yolo_dataset_manager/requirements.txt.")
    selected = list(dict.fromkeys(str(item).strip() for item in chart_ids if str(item).strip()))
    if not selected:
        raise ValueError("Select at least one chart.")

    images_root = images_dir.resolve()
    labels_root = (labels_dir or images_dir).resolve()
    scan = scan_dataset(images_root, labels_root)
    paired_items = scan["paired"]
    image_key_map: Dict[str, str] = dict(scan["image_key_map"])
    label_key_map: Dict[str, str] = dict(scan["label_key_map"])
    class_names = clean_classes(classes)

    output_root: Path | None = None
    if output_dir is not None:
        output_root = output_dir.resolve()
        output_root.mkdir(parents=True, exist_ok=True)

    class_counter: Counter[int] = Counter()
    objects_per_image: List[int] = []
    bbox_w: List[float] = []
    bbox_h: List[float] = []
    bbox_area: List[float] = []
    image_widths: List[int] = []
    image_heights: List[int] = []
    parse_errors = 0
    unreadable_images = 0

    for pair in paired_items:
        key = str(pair["key"])
        image_rel = image_key_map.get(key)
        label_rel = label_key_map.get(key)
        if not image_rel or not label_rel:
            continue
        image_abs = ensure_in_base(images_root, (images_root / image_rel).resolve())
        label_abs = ensure_in_base(labels_root, (labels_root / label_rel).resolve())
        if not image_abs.exists() or not label_abs.exists():
            continue

        try:
            with Image.open(image_abs) as image:
                width, height = image.size
            image_widths.append(int(width))
            image_heights.append(int(height))
        except (OSError, UnidentifiedImageError):
            unreadable_images += 1

        rows = parse_yolo_label_file(label_abs)
        objects_per_image.append(len(rows))
        for row in rows:
            class_id = int(row["class_id"])
            class_counter[class_id] += 1
            width = float(row["width"])
            height = float(row["height"])
            area = float(row["area"])
            bbox_w.append(width)
            bbox_h.append(height)
            bbox_area.append(area)
        if not rows and label_abs.stat().st_size > 0:
            parse_errors += 1

    charts: List[Dict[str, Any]] = []

    for chart_id in selected:
        if chart_id == "class_distribution":
            fig, ax = plt.subplots(figsize=(9, 5))
            if class_counter:
                sorted_items = sorted(class_counter.items(), key=lambda item: item[0])
                labels = []
                values = []
                for class_id, count in sorted_items:
                    class_label = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
                    labels.append(f"{class_id}:{class_label}")
                    values.append(count)
                ax.bar(labels, values, color="#0f766e")
                ax.tick_params(axis="x", rotation=45, labelsize=8)
                ax.set_ylabel("Objects")
            else:
                ax.text(0.5, 0.5, "No objects found", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_title("Class Distribution")
            ax.grid(axis="y", alpha=0.2)
            charts.append(_build_chart_payload(fig, chart_id, "Class Distribution", output_root))
            continue

        if chart_id == "objects_per_image":
            fig, ax = plt.subplots(figsize=(8, 5))
            if objects_per_image:
                bins = min(20, max(5, max(objects_per_image) + 1))
                ax.hist(objects_per_image, bins=bins, color="#1f7a8c", edgecolor="white")
            else:
                ax.text(0.5, 0.5, "No labels found", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_title("Objects Per Image")
            ax.set_xlabel("Objects")
            ax.set_ylabel("Images")
            ax.grid(axis="y", alpha=0.2)
            charts.append(_build_chart_payload(fig, chart_id, "Objects Per Image", output_root))
            continue

        if chart_id == "bbox_wh":
            fig, ax = plt.subplots(figsize=(7, 7))
            if bbox_w and bbox_h:
                ax.scatter(bbox_w, bbox_h, s=8, alpha=0.35, color="#0b7285")
                ax.set_xlim(0, max(1.0, max(bbox_w) * 1.05))
                ax.set_ylim(0, max(1.0, max(bbox_h) * 1.05))
            else:
                ax.text(0.5, 0.5, "No boxes found", ha="center", va="center")
            ax.set_title("BBox Width vs Height (normalized)")
            ax.set_xlabel("Width")
            ax.set_ylabel("Height")
            ax.grid(alpha=0.2)
            charts.append(_build_chart_payload(fig, chart_id, "BBox Width vs Height", output_root))
            continue

        if chart_id == "bbox_area":
            fig, ax = plt.subplots(figsize=(8, 5))
            if bbox_area:
                ax.hist(bbox_area, bins=40, color="#f4a261", edgecolor="white")
            else:
                ax.text(0.5, 0.5, "No boxes found", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_title("BBox Area Distribution (normalized)")
            ax.set_xlabel("Area")
            ax.set_ylabel("Boxes")
            ax.grid(axis="y", alpha=0.2)
            charts.append(_build_chart_payload(fig, chart_id, "BBox Area Distribution", output_root))
            continue

        if chart_id == "pair_coverage":
            fig, ax = plt.subplots(figsize=(6, 6))
            paired_count = int(scan["pair_count"])
            image_only = len(scan["images_without_labels"])
            label_only = len(scan["labels_without_images"])
            values = [paired_count, image_only, label_only]
            labels = [f"paired ({paired_count})", f"images_only ({image_only})", f"labels_only ({label_only})"]
            if sum(values) <= 0:
                ax.text(0.5, 0.5, "No files found", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.pie(
                    values,
                    labels=labels,
                    autopct="%1.1f%%",
                    colors=["#2a9d8f", "#e9c46a", "#e76f51"],
                    startangle=90,
                )
                ax.axis("equal")
            ax.set_title("Pair Coverage")
            charts.append(_build_chart_payload(fig, chart_id, "Pair Coverage", output_root))
            continue

        if chart_id == "image_sizes":
            fig, ax = plt.subplots(figsize=(8, 6))
            if image_widths and image_heights:
                ax.scatter(image_widths, image_heights, s=10, alpha=0.45, color="#5e548e")
            else:
                ax.text(0.5, 0.5, "No readable images", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_title("Image Size Distribution")
            ax.set_xlabel("Width (px)")
            ax.set_ylabel("Height (px)")
            ax.grid(alpha=0.2)
            charts.append(_build_chart_payload(fig, chart_id, "Image Size Distribution", output_root))
            continue

    paired_count = int(scan["pair_count"])
    image_count = int(scan["image_count"])
    label_count = int(scan["label_count"])
    coverage = {
        "paired_ratio_from_images": _safe_ratio(paired_count, image_count),
        "paired_ratio_from_labels": _safe_ratio(paired_count, label_count),
    }

    return {
        "output_dir": str(output_root) if output_root else None,
        "chart_count": len(charts),
        "charts": charts,
        "summary": {
            "image_count": image_count,
            "label_count": label_count,
            "paired_count": paired_count,
            "images_without_labels": len(scan["images_without_labels"]),
            "labels_without_images": len(scan["labels_without_images"]),
            "objects_total": int(sum(class_counter.values())),
            "classes_found": len(class_counter),
            "parse_error_files": parse_errors,
            "unreadable_images": unreadable_images,
            "coverage": coverage,
        },
    }
