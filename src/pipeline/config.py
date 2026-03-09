import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any


@dataclass
class SegmentationConfig:
    """Configuration for the Cellpose segmentation subpipeline."""
    enabled: bool = True
    patch_size: tuple = (224, 224)
    stride: tuple = (192, 192)
    overlap_strategy: str = 'iou_matching'
    iou_threshold: float = 0.5
    gpu: bool = True
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    simplify_tolerance: float = 1.0
    save_patches: bool = False
    presegment_tissue: bool = False

@dataclass
class PipelineConfig:
    """Configuration for the pre-processing pipeline."""
    annotations_dir: str
    wsi_dir: str
    output_dir: str
    interim_dir: str = ""  # Directory to store intermediate cleaned annotations
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)

    def __post_init__(self):
        self.annotations_dir = str(Path(self.annotations_dir).resolve())
        self.wsi_dir = str(Path(self.wsi_dir).resolve())
        self.output_dir = str(Path(self.output_dir).resolve())
        
        if not self.interim_dir:
            self.interim_dir = str(Path(self.output_dir) / "interim_clean_annotations")
        else:
            self.interim_dir = str(Path(self.interim_dir).resolve())
        
        # Ensure output directories exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.interim_dir).mkdir(parents=True, exist_ok=True)


class PipelineState:
    """Class to track the state and statistics of the pipeline run."""
    def __init__(self):
        # Stage 1: Deduplication
        self.total_annotations_found: int = 0
        self.unique_annotations: int = 0
        self.duplicates: Dict[str, List[str]] = {}
        
        # Stage 2: Label Standardization
        self.label_counts: Dict[str, int] = {}
        self.unknown_labels: Dict[str, int] = {}
        
        # Stage 3: Region Extraction
        self.matched_wsis: int = 0
        self.unmatched_annotations: List[str] = []
        self.extracted_regions: List[Dict[str, Any]] = []
        self.extraction_errors: List[Dict[str, Any]] = []
        
        # Stage 4: Segmentation
        self.segmentation_results: List[Dict[str, Any]] = []
        self.segmentation_errors: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deduplication": {
                "total_annotations_found": self.total_annotations_found,
                "unique_annotations": self.unique_annotations,
                "duplicates": self.duplicates,
            },
            "label_standardization": {
                "label_counts": self.label_counts,
                "unknown_labels": self.unknown_labels,
            },
            "region_extraction": {
                "matched_wsis": self.matched_wsis,
                "unmatched_annotations": self.unmatched_annotations,
                "extracted_regions_count": len(self.extracted_regions),
                "extracted_regions": self.extracted_regions,
                "extraction_errors": self.extraction_errors,
            },
            "segmentation": {
                "processed_regions": len(self.segmentation_results),
                "results": self.segmentation_results,
                "errors": self.segmentation_errors
            }
        }

    def save_report(self, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)
