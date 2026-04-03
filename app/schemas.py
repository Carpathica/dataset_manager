from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SessionRequest(BaseModel):
    images_dir: str
    labels_dir: Optional[str] = None
    classes_file: Optional[str] = None


class OrphanActionItem(BaseModel):
    kind: Literal["image", "label"]
    rel_path: str
    action: Literal["keep", "delete"]


class OrphanActionsRequest(BaseModel):
    items: List[OrphanActionItem] = Field(default_factory=list)


class MoveLabelsRequest(BaseModel):
    target_labels_dir: Optional[str] = None
    move_unpaired: bool = False
    mode: Literal["move", "copy"] = "move"


class SplitRequest(BaseModel):
    output_dir: str
    train_ratio: float = Field(default=0.7, ge=0.0)
    val_ratio: float = Field(default=0.2, ge=0.0)
    test_ratio: float = Field(default=0.1, ge=0.0)
    seed: int = 42
    mode: Literal["copy", "move"] = "copy"
    include_only_paired: bool = True
    generate_yaml: bool = False
    yaml_name: str = "data.yaml"
    classes: List[str] = Field(default_factory=list)


class MergePreviewRequest(BaseModel):
    dataset_dirs: List[str] = Field(default_factory=list)


class MergeDuplicateResolution(BaseModel):
    group_id: str
    keep_index: int = Field(default=0, ge=0)


class PairDuplicateResolution(BaseModel):
    group_id: str
    keep_index: int = Field(default=0, ge=0)


class PairDuplicatesResolveRequest(BaseModel):
    resolutions: List[PairDuplicateResolution] = Field(default_factory=list)


class MergeRunRequest(BaseModel):
    dataset_dirs: List[str] = Field(default_factory=list)
    output_dir: str
    mode: Literal["copy", "move"] = "copy"
    resolutions: List[MergeDuplicateResolution] = Field(default_factory=list)
    delete_skipped_duplicates: bool = False
    generate_yaml: bool = False
    yaml_name: str = "data.yaml"
    classes: List[str] = Field(default_factory=list)


class AnalyticsRequest(BaseModel):
    charts: List[
        Literal[
            "class_distribution",
            "objects_per_image",
            "bbox_wh",
            "bbox_area",
            "pair_coverage",
            "image_sizes",
        ]
    ] = Field(default_factory=list)
    output_dir: Optional[str] = None
