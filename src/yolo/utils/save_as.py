from geojson import Feature, FeatureCollection, Polygon
from shapely.geometry import box, mapping
import json
import numpy as np
from tqdm import tqdm

# source: supervision
def polygon_to_xyxy(polygon: np.ndarray):
    x_min, y_min = np.min(polygon, axis=0)
    x_max, y_max = np.max(polygon, axis=0)
    return np.array([x_min, y_min, x_max, y_max])

def yolo_to_geojson(label_file, image_width, image_height, output_file="bboxes.geojson"):
    features = []

    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Label file {label_file} not found.")
        return
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue

        try:
            class_id, center_x, center_y, width, height = map(float, line.strip().split())
        except ValueError:
            print(f"Warning: Invalid format in line {i+1}: {line.strip()}. Skipping.")
            continue

        center_x_px = center_x * image_width
        center_y_px = center_y * image_height
        width_px = width * image_width
        height_px = height * image_height

        x_min = center_x_px - (width_px / 2)
        x_max = center_x_px + (width_px / 2)
        y_min = center_y_px - (height_px / 2)
        y_max = center_y_px + (height_px / 2)
        
        # ensure coordinates are within image bounds
        x_min = max(0, min(x_min, image_width))
        x_max = max(0, min(x_max, image_width))
        y_min = max(0, min(y_min, image_height))
        y_max = max(0, min(y_max, image_height))
        
        # polygon (rectangle) for the bounding box
        # geojson uses [x, y] format
        coords = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
            [x_min, y_min]
        ]

        polygon = Polygon([coords])

        feature = Feature(
            geometry=polygon,
            properties={
                "id": i + 1,
                "class_id": int(class_id),
                "name": f"Object_{i+1}_Class_{int(class_id)}"
            }
        )
        features.append(feature)
    
    feature_collection = FeatureCollection(features)

    try:
        with open(output_file, "w") as f:
            json.dump(feature_collection, f, indent=2)
        print(f"GeoJSON file saved to {output_file}")
    except Exception as e:
        print(f"Error saving GeoJSON file: {e}")

def npy_mask_to_geojson(mask_file, output_file):
    mask = np.load(mask_file)
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, min={mask.min()}, max={mask.max()}")

    # unique instances -> we assume each instance has a different value
    unique_values = np.unique(mask)
    print(f"Found {len(unique_values)} unique values")

    unique_instances = unique_values[unique_values > 0] # 0 for background
    features = []

    for i, value in enumerate(tqdm(unique_instances, desc="Processing instances")):
        # get all pixels belonging to this instance (account for float inconsistencies)
        instance_mask = (mask == value)
        rows, cols = np.where(instance_mask)

        if len(rows) == 0:
            continue  # skip empty masks (shouldn’t happen, but safe)

        x_min, x_max = cols.min(), cols.max()
        y_min, y_max = rows.min(), rows.max()

        rect = box(float(x_min), float(y_min), float(x_max), float(y_max))

        features.append({
            "type": "Feature",
            "properties": {
                "id": int(i),
                "mask_value": float(value)
            },
            "geometry": mapping(rect)
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_file, "w") as f:
        json.dump(geojson, f, indent=2)


def main():
    # label_file = "/Users/simon/Downloads/kaggle 3/working/yolo_dataset/labels/train/tile_0013_640_1280.txt"
    # image_width = 640
    # image_height = 640
    # yolo_to_geojson(label_file, image_width, image_height, output_file="/Users/simon/Downloads/tile_0013_640_1280.geojson")
    mask_file = '/Users/simon/Downloads/stitched_mask (1).npy'
    npy_mask_to_geojson(mask_file, output_file='/Users/simon/Downloads/stitched_mask.geojson')


if __name__ == "__main__":
    main()