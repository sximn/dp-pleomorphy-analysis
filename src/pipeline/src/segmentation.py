from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from cellpose import models
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.utils import misc
from tiatoolbox import logger
from scipy import ndimage
import pickle
import gc
import torch
import cv2
import json
from geojson import Feature, FeatureCollection, Polygon
from typing import List


class SegmentationSlidingWindowExtractor(SlidingWindowPatchExtractor):

    def _generate_location_df(self):
        slide_dimension = self.wsi.slide_dimensions(self.resolution, self.units)
        image_shape = (slide_dimension[0], slide_dimension[1])
        patch_input_shape = self.patch_size
        stride_shape = self.stride

        coord_list = self.get_coordinates(
            image_shape=image_shape,
            patch_input_shape=patch_input_shape,
            stride_shape=stride_shape,
            input_within_bound=False,
        )

        # adjust coordinates for boundary patches
        adjusted_coords = []
        for coord in coord_list:
            x_start, y_start, x_end, y_end = coord
            if x_end > image_shape[0]:
                x_start = max(0, image_shape[0] - patch_input_shape[0])
                x_end = x_start + patch_input_shape[0]
            if y_end > image_shape[1]:
                y_start = max(0, image_shape[1] - patch_input_shape[1])
                y_end = y_start + patch_input_shape[1]
            adjusted_coords.append([x_start, y_start, x_end, y_end])

        self.coordinate_list = np.array(adjusted_coords)

        # filter coordinates based on mask
        if self.mask is not None:
            selected_coord_indices = self.filter_coordinates(
                self.mask,
                self.coordinate_list,
                wsi_shape=image_shape,
                min_mask_ratio=self.min_mask_ratio,
            )
            self.coordinate_list = self.coordinate_list[selected_coord_indices]
            if len(self.coordinate_list) == 0:
                logger.warning(
                    "No candidate coordinates left after filtering.",
                    stacklevel=2,
                )

        self.locations_df = misc.read_locations(input_table=self.coordinate_list[:, :2])
        return self
    
    def __iter__(self):
        for idx in range(len(self.coordinate_list)):
            patch = self[idx]
            coords = self.coordinate_list[idx]
            yield idx, patch, coords


class NucleiSegmentationPipeline:

    def __init__(self, wsi_path, patch_size=(640, 640), stride=(640, 640),
                 overlap_strategy='iou_matching', gpu=True, iou_threshold=0.5,
                 flow_threshold=0.4, cellprob_threshold=0.0):
        self.wsi_path = wsi_path
        self.wsi = WSIReader.open(input_img=wsi_path)
        self.patch_size = patch_size
        self.stride = stride
        self.overlap_strategy = overlap_strategy
        self.iou_threshold = iou_threshold

        self.model = models.CellposeModel(gpu=gpu)

        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.tile_norm_blocksize = 0

    def segment_patch(self, patch):
        img_selected_channels = patch.copy()

        masks, flows, styles = self.model.eval(
            img_selected_channels,
            batch_size=32,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            normalize={"tile_norm_blocksize": self.tile_norm_blocksize}
        )
        return masks, flows

    def stitch_masks_iou_matching(self, patch_masks_dict, wsi_shape):
        print(f"Stitching {len(patch_masks_dict)} patches using IoU matching (threshold={self.iou_threshold})")
        
        stitched_mask = np.zeros(wsi_shape, dtype=np.int32)
        current_max_id = 0
        
        sorted_items = sorted(patch_masks_dict.items())
        
        for idx, (mask, coords) in tqdm(sorted_items, desc="Stitching with IoU matching"):
            x_start, y_start, x_end, y_end = coords

            actual_h, actual_w = mask.shape[:2]
            y_end_actual = y_start + actual_h
            x_end_actual = x_start + actual_w

            existing_region = stitched_mask[y_start:y_end_actual, x_start:x_end_actual].copy()
            unique_labels = np.unique(mask[mask > 0])
            
            for local_id in unique_labels:
                cell_mask = (mask == local_id)
                overlapping_ids = np.unique(existing_region[cell_mask])
                overlapping_ids = overlapping_ids[overlapping_ids > 0]
                
                matched = False
                best_iou = 0
                best_existing_id = None
                
                for existing_id in overlapping_ids:
                    existing_cell = (existing_region == existing_id)
                    intersection = np.sum(cell_mask & existing_cell)
                    union = np.sum(cell_mask | existing_cell)
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_existing_id = existing_id
                
                if best_iou > self.iou_threshold:
                    stitched_mask[y_start:y_end_actual, x_start:x_end_actual][cell_mask] = best_existing_id
                    matched = True
                
                if not matched:
                    current_max_id += 1
                    stitched_mask[y_start:y_end_actual, x_start:x_end_actual][cell_mask] = current_max_id
        
        return stitched_mask

    def stitch_masks(self, patch_masks_dict, wsi_shape):
        if self.overlap_strategy == 'iou_matching':
            return self.stitch_masks_iou_matching(patch_masks_dict, wsi_shape)

        print(f"Stitching {len(patch_masks_dict)} patches into WSI shape {wsi_shape}")
        stitched_mask = np.zeros(wsi_shape, dtype=np.int32)
        current_max_id = 0
        sorted_items = sorted(patch_masks_dict.items())
        
        for idx, (mask, coords) in tqdm(sorted_items, desc="Stitching masks"):
            x_start, y_start, x_end, y_end = coords
            actual_h, actual_w = mask.shape[:2]
            y_end_actual = y_start + actual_h
            x_end_actual = x_start + actual_w

            if mask.max() > 0:
                unique_labels = np.unique(mask[mask > 0])
                relabeled_mask = np.zeros_like(mask)
                for old_label in unique_labels:
                    current_max_id += 1
                    relabeled_mask[mask == old_label] = current_max_id
                mask = relabeled_mask
            
            if self.overlap_strategy == 'first':
                region = stitched_mask[y_start:y_end_actual, x_start:x_end_actual]
                region[region == 0] = mask[region == 0]

        return stitched_mask
    
    def run_full_pipeline(self, output_dir, save_patches=False, resolution=0, units="level",
                          presegment_tissue=True):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        extractor = SegmentationSlidingWindowExtractor(
            input_img=self.wsi,
            patch_size=self.patch_size,
            stride=self.stride,
            within_bound=True,
            resolution=resolution,
            units=units,
            input_mask="otsu" if presegment_tissue else None,
            min_mask_ratio=0.5 if presegment_tissue else 0.0,
        )
        
        slide_dimension = self.wsi.slide_dimensions(resolution, units)
        wsi_shape = (slide_dimension[1], slide_dimension[0])

        print(f"WSI shape (H, W): {wsi_shape}")
        print(f"Processing {len(extractor.coordinate_list)} patches...")
        
        patch_masks_dict = {}

        for idx, patch, coords in tqdm(extractor, desc="Processing patches"):
            mask, flows = self.segment_patch(patch)
            del flows
            patch_masks_dict[idx] = (mask, coords)
            
            if save_patches:
                patch_dir = output_dir / "patches"
                patch_dir.mkdir(exist_ok=True)
                xs, ys, xe, ye = coords
                id_coord_label = f"{idx:04d}_XYXY_{xs}_{ys}_{xe}_{ye}"
                cv2.imwrite(str(patch_dir / f"patch_{id_coord_label}.png"), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                np.save(str(patch_dir / f"mask_{id_coord_label}.npy"), mask)

            del patch
            torch.cuda.empty_cache()
            gc.collect()
            
        print("\nStitching masks...")
        stitched_mask = self.stitch_masks(patch_masks_dict, wsi_shape)
        return stitched_mask


def npy_mask_to_geojson_polygon(mask, output_file, simplify_tolerance=1.0):
    """
    Extract precise polygon contours from instance mask and save to GeoJSON.
    """
    unique_values = np.unique(mask)
    unique_instances = unique_values[unique_values > 0]
    features = []

    for i, value in enumerate(tqdm(unique_instances, desc="Converting instances to Polygons")):
        instance_mask = (mask == value).astype(np.uint8)
        if instance_mask.sum() == 0:
            continue

        contours, _ = cv2.findContours(
            instance_mask, 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            continue

        contour = max(contours, key=cv2.contourArea)
        polygon_coords = contour.squeeze()

        if polygon_coords.ndim == 1 or len(polygon_coords) < 3:
            continue

        polygon_coords = polygon_coords.tolist()
        polygon_coords = [[float(x), float(y)] for x, y in polygon_coords]

        if simplify_tolerance > 0:
            from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
            if polygon_coords[0] != polygon_coords[-1]:
                polygon_coords_closed = polygon_coords + [polygon_coords[0]]
            else:
                polygon_coords_closed = polygon_coords

            try:
                poly = ShapelyPolygon(polygon_coords_closed)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if isinstance(poly, MultiPolygon):
                    poly = max(poly.geoms, key=lambda p: p.area)

                poly_simplified = poly.simplify(simplify_tolerance, preserve_topology=True)

                if isinstance(poly_simplified, MultiPolygon):
                    poly_simplified = max(poly_simplified.geoms, key=lambda p: p.area)
                if hasattr(poly_simplified, 'exterior'):
                    polygon_coords = list(poly_simplified.exterior.coords)
            except Exception as e:
                pass

        if len(polygon_coords) < 3:
            continue
        if polygon_coords[0] != polygon_coords[-1]:
            polygon_coords.append(polygon_coords[0])
        if len(polygon_coords) < 4: 
            continue

        polygon = Polygon([polygon_coords])
        feature = Feature(
            geometry=polygon,
            properties={
                "id": int(i),
                "mask_value": float(value),
                "area": float(instance_mask.sum()),
                "perimeter": float(cv2.arcLength(contour, True)),
                "classification": { "name": "nuclei", "color": [0, 255, 0] }
            }
        )
        features.append(feature)

    feature_collection = FeatureCollection(features)
    with open(output_file, "w") as f:
        json.dump(feature_collection, f, indent=2)


def run_segmentation(input_regions: List[str], output_dir: str, config, state=None):
    """
    Runs the segmentation subpipeline on a list of extracted regions (.ome.tif).
    Converts segmented masks into GeoJSON and records stats into pipeline state.
    """
    if not config.enabled:
        print("Segmentation is disabled via config.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Stage 4: Nuclei Segmentation ---")
    for region_path in tqdm(input_regions, desc="Segmenting Extracted Regions"):
        path_obj = Path(region_path)
        try:
            print(f"\nProcessing {path_obj.name}")
            pipeline = NucleiSegmentationPipeline(
                wsi_path=region_path,
                patch_size=config.patch_size,
                stride=config.stride,
                overlap_strategy=config.overlap_strategy,
                iou_threshold=config.iou_threshold,
                gpu=config.gpu,
                flow_threshold=config.flow_threshold,
                cellprob_threshold=config.cellprob_threshold
            )
            
            # Use interim directory inside output_dir for holding array masks or patches
            region_out_dir = Path(output_dir) / f"{path_obj.stem}_seg_temp"
            
            stitched_mask = pipeline.run_full_pipeline(
                output_dir=str(region_out_dir),
                save_patches=config.save_patches,
                resolution=0,
                units="level",
                presegment_tissue=config.presegment_tissue,
            )
            
            # Convert mask directly to geojson
            output_geojson = Path(output_dir) / f"{path_obj.stem}_segmentation.geojson"
            npy_mask_to_geojson_polygon(
                stitched_mask, 
                output_file=str(output_geojson),
                simplify_tolerance=config.simplify_tolerance
            )
            
            # Clean up temp files if not saving patches
            if not config.save_patches:
                import shutil
                shutil.rmtree(str(region_out_dir), ignore_errors=True)
                
            if state:
                state.segmentation_results.append({
                    "region": region_path,
                    "geojson": str(output_geojson),
                    "cells_segmented": len(np.unique(stitched_mask)) - 1
                })
                
        except Exception as e:
            print(f"Error segmenting {path_obj.name}: {e}")
            if state:
                state.segmentation_errors.append({
                    "region": region_path,
                    "error": str(e)
                })

