import argparse
from pathlib import Path

# Important: ensure sys.path includes the src so imports work natively
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from config import PipelineConfig, PipelineState
from duplicated_annotations import get_deduped_filepaths
from label_standardization import standardize_labels
from region_extractor import extract_regions_for_annotations
from segmentation import run_segmentation

def run_pipeline(config: PipelineConfig):
    print("=========================================")
    print("Starting Pre-processing Pipeline")
    print("=========================================")
    print(f"Annotations Dir: {config.annotations_dir}")
    print(f"WSI Dir:         {config.wsi_dir}")
    print(f"Output Dir:      {config.output_dir}")
    print(f"Interim Dir:     {config.interim_dir}")
    if config.skip_annotations:
        print(f"Skipping Annots: {config.skip_annotations}")
    print(f"Segmentation:    {'Enabled' if config.segmentation.enabled else 'Disabled'}")
    
    state = PipelineState()
    
    try:
        # Stage 1: Deduplicate target geojson files
        unique_files = get_deduped_filepaths(
            config.annotations_dir, 
            state=state, 
            skip_annotations=config.skip_annotations
        )
        
        # Stage 2: Standardize labels inside geojsons
        clean_files = standardize_labels(unique_files, config.interim_dir, state=state)
        
        # Stage 3: Extract image regions for each clean geojson
        extract_regions_for_annotations(clean_files, config.wsi_dir, config.output_dir, state=state)
        
        # Stage 4: Run instance segmentation on extracted image regions
        run_segmentation(state.extracted_regions, config.output_dir, config.segmentation, state=state)
        
    except Exception as e:
        print(f"\nPipeline Failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Save pipeline statistics to report
        report_path = Path(config.output_dir) / 'pipeline_stats.json'
        state.save_report(str(report_path))
        print(f"\nPipeline report saved to: {report_path}")
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pleomorphy pre-processing pipeline.")
    parser.add_argument("--annotations_dir", required=True, help="Directory containing input geojson annotations.")
    parser.add_argument("--wsi_dir", required=True, help="Directory containing WSI (.mrxs) files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save final OME-TIFFs and report.")
    parser.add_argument("--interim_dir", default="", help="Directory to save interim clean geojsons.")
    parser.add_argument("--skip_annotations", nargs="*", default=[], help="List of annotation file names to skip.")
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        annotations_dir=args.annotations_dir,
        wsi_dir=args.wsi_dir,
        output_dir=args.output_dir,
        interim_dir=args.interim_dir,
        skip_annotations=args.skip_annotations
    )
    
    run_pipeline(config)
