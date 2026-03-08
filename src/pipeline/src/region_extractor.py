import json
import os
import numpy as np
from large_image import getTileSource
import tifffile
import openslide
from shapely.geometry import Polygon, mapping
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import List

def load_geojson(geojson_path):
    """Load geojson file containing annotations."""
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    return geojson_data


def find_bounding_rectangle(geojson_data):
    """
    Find the bounding rectangle that contains all annotations.
    Returns: (min_x, min_y, max_x, max_y) - the coordinates of the rectangle
    """
    all_points = []
    
    # Extract all points from all features
    for feature in geojson_data.get('features', []):
        geometry = feature.get('geometry', {})
        geometry_type = geometry.get('type', '')
        coordinates = geometry.get('coordinates', [])
        
        if geometry_type == 'Polygon':
            # For polygons, coordinates are [exterior_ring, interior_ring1, ...]
            for ring in coordinates:
                all_points.extend(ring)
        elif geometry_type == 'MultiPolygon':
            # For multipolygons, coordinates are [polygon1, polygon2, ...]
            for polygon in coordinates:
                for ring in polygon:
                    all_points.extend(ring)
        elif geometry_type == 'LineString':
            all_points.extend(coordinates)
        elif geometry_type == 'Point':
            all_points.append(coordinates)
    
    if not all_points:
        raise ValueError("No valid geometries found in the geojson file")
    
    # Convert to numpy array for easier operations
    points = np.array(all_points)
    
    # Get min and max coordinates
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    
    return min_x, min_y, max_x, max_y


def remap_annotations(geojson_data, offset_x, offset_y):
    """
    Remap annotations coordinates by subtracting the offset.
    This shifts all annotations to be relative to the top-left of the extracted region.
    """
    remapped_geojson = geojson_data.copy()
    
    for feature in remapped_geojson.get('features', []):
        geometry = feature.get('geometry', {})
        geometry_type = geometry.get('type', '')
        coordinates = geometry.get('coordinates', [])
        
        if geometry_type == 'Polygon':
            for i, ring in enumerate(coordinates):
                remapped_ring = []
                for point in ring:
                    remapped_ring.append([point[0] - offset_x, point[1] - offset_y])
                coordinates[i] = remapped_ring
        
        elif geometry_type == 'MultiPolygon':
            for i, polygon in enumerate(coordinates):
                remapped_polygon = []
                for ring in polygon:
                    remapped_ring = []
                    for point in ring:
                        remapped_ring.append([point[0] - offset_x, point[1] - offset_y])
                    remapped_polygon.append(remapped_ring)
                coordinates[i] = remapped_polygon
        
        elif geometry_type == 'LineString':
            remapped_line = []
            for point in coordinates:
                remapped_line.append([point[0] - offset_x, point[1] - offset_y])
            feature['geometry']['coordinates'] = remapped_line
        
        elif geometry_type == 'Point':
            feature['geometry']['coordinates'] = [
                coordinates[0] - offset_x,
                coordinates[1] - offset_y
            ]
    
    return remapped_geojson


def extract_region_from_wsi(wsi_path, output_path, x_min, y_min, width, height, level=0):
    """
    Extract a region from the WSI image and save it using large_image.
    
    Args:
        wsi_path: Path to the WSI file (.mrxs format)
        output_path: Path to save the extracted region (as .ome.tif)
        x_min, y_min: Top-left coordinates of the region to extract
        width, height: Width and height of the region to extract
        level: Not used with large_image, but kept for compatibility
    """
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Open the WSI file with large_image
    source = getTileSource(str(wsi_path))
    
    # Get metadata for resolution information
    metadata = source.getMetadata()
    mpp_x = float(metadata.get("mm_x", 0)) * 1000  # convert mm to microns
    mpp_y = float(metadata.get("mm_y", 0)) * 1000
    
    # Define the region to extract
    region = {
        'left': int(x_min),
        'top': int(y_min),
        'width': int(width),
        'height': int(height),
        'units': 'base_pixels'
    }
    
    # Get the region as a PIL image
    tile_image, _ = source.getRegion(region=region, format='PIL')
    
    # Convert to RGB numpy array
    tile_rgb = np.array(tile_image.convert("RGB"))
    
    # Save as OME-TIFF with resolution and compression
    tifffile.imwrite(
        output_path,
        tile_rgb,
        photometric='rgb',
        tile=(256, 256),  # tiled like WSIs
        compression='deflate',
        resolution=(1 / mpp_x, 1 / mpp_y) if mpp_x > 0 and mpp_y > 0 else None,
        resolutionunit='CENTIMETER',
        metadata={'axes': 'YXS'},
        ome=True
    )
    return output_path


def match_wsi_file(geojson_name: str, wsi_dir: str) -> str:
    """Find the corresponding WSI file for a given geojson annotation based on filename."""
    wsi_path_obj = Path(wsi_dir)
    # Search for all mrxs files in the WSI directory
    mrxs_files = list(wsi_path_obj.glob("*.mrxs"))
    
    for mrxs_file in mrxs_files:
        # e.g., if mrxs_file.stem is 'slide-2024-04-03T07-52-35-R1-S2' and
        # geojson_name is 'slide-2024-04-03T07-52-35-R1-S2 poleA.geojson',
        # the mrxs_file stem covers the geojson name.
        if mrxs_file.stem in geojson_name:
            return str(mrxs_file)
    return None

def extract_regions_for_annotations(clean_geojson_files: List[str], wsi_dir: str, output_dir: str, state=None):
    """
    Extracts regions for each standardized geojson file from its corresponding WSI.
    Populates statistics in the state object if provided.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Stage 3: Region Extraction ---")
    for geojson_path in tqdm(clean_geojson_files, desc="Extracting Regions"):
        file_path = Path(geojson_path)
        geojson_name = file_path.name
        
        wsi_path = match_wsi_file(geojson_name, wsi_dir)
        if not wsi_path:
            print(f"Warning: No matching WSI found for {geojson_name}")
            if state:
                state.unmatched_annotations.append(geojson_name)
            continue
            
        if state:
            state.matched_wsis += 1
            
        try:
            geojson_data = load_geojson(geojson_path)
            
            # Find boundaries
            min_x, min_y, max_x, max_y = find_bounding_rectangle(geojson_data)
            width = max_x - min_x
            height = max_y - min_y
            
            # Remap
            remapped_geojson = remap_annotations(geojson_data, min_x, min_y)
            remapped_geojson_name = f'remapped_{file_path.name}'
            remapped_geojson_path = Path(output_dir) / remapped_geojson_name
            with open(remapped_geojson_path, 'w', encoding='utf-8') as f:
                json.dump(remapped_geojson, f, indent=2)
                
            # Extract Region
            output_wsi_name = f'remapped_{Path(wsi_path).stem}_{file_path.stem}.ome.tif'
            output_wsi_path = Path(output_dir) / output_wsi_name
            
            extract_region_from_wsi(wsi_path, str(output_wsi_path), min_x, min_y, width, height, level=0)
            
            if state:
                state.extracted_regions.append(str(output_wsi_path))
                
        except Exception as e:
            print(f"Error extracting region for {geojson_name}: {e}")
            if state:
                state.extraction_errors.append({
                    "geojson": geojson_name,
                    "wsi": wsi_path,
                    "error": str(e)
                })

