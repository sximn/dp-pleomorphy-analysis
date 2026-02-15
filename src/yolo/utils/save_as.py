from geojson import Feature, FeatureCollection, Polygon
from shapely.geometry import box, mapping, shape
import json
import numpy as np
from tqdm import tqdm
import cv2

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

def npy_mask_to_geojson_bbox(mask_file, output_file):
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
            continue  # skip empty masks (shouldnt happen, but safe)

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
    print(f"GeoJSON with bounding boxes saved to {output_file}")


def npy_mask_to_geojson_polygon(mask_file, output_file, simplify_tolerance=1.0):
    """
    Extract precise polygon contours from instance mask.
    
    Args:
        mask_file: Path to .npy mask file
        output_file: Path to output GeoJSON file
        simplify_tolerance: Tolerance for polygon simplification, higher values = simpler polygons. Set to 0 to disable simplification.
    """
    mask = np.load(mask_file)
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, min={mask.min()}, max={mask.max()}")

    image_height = mask.shape[0]

    # unique instances -> we assume each instance has a different value
    unique_values = np.unique(mask)
    print(f"Found {len(unique_values)} unique values")

    unique_instances = unique_values[unique_values > 0]  # 0 for background
    features = []

    for i, value in enumerate(tqdm(unique_instances, desc="Processing instances")):
        # create binary mask for this instance
        instance_mask = (mask == value).astype(np.uint8)

        if instance_mask.sum() == 0:
            continue

        # find contours
        contours, hierarchy = cv2.findContours(
            instance_mask, 
            cv2.RETR_EXTERNAL,  # only external contours
            cv2.CHAIN_APPROX_SIMPLE  # compress horizontal, vertical, and diagonal segments
        )

        if len(contours) == 0:
            continue

        # use the largest contour if multiple exist (shouldnt happen with instance masks)
        contour = max(contours, key=cv2.contourArea)

        # convert contour to polygon coordinates
        # openCV contours are in format: [[[x, y]], [[x, y]], ...]
        # we need: [[x, y], [x, y], ...]
        polygon_coords = contour.squeeze()

        # handle edge cases
        if polygon_coords.ndim == 1:
            # single point, skip
            continue

        if len(polygon_coords) < 3:
            # contour too small to form a polygon, skip
            continue
        polygon_coords = polygon_coords.tolist()

        polygon_coords = [[float(x), float(y)] for x, y in polygon_coords]

        if simplify_tolerance > 0:
            from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
            # need to close the polygon for Shapely
            if polygon_coords[0] != polygon_coords[-1]:
                polygon_coords_closed = polygon_coords + [polygon_coords[0]]
            else:
                polygon_coords_closed = polygon_coords

            try:
                poly = ShapelyPolygon(polygon_coords_closed)
                if not poly.is_valid:
                    # try to fix invalid polygon
                    poly = poly.buffer(0)

                # check if buffer resulted in MultiPolygon
                if isinstance(poly, MultiPolygon):
                    # use the largest polygon from the MultiPolygon
                    poly = max(poly.geoms, key=lambda p: p.area)

                poly_simplified = poly.simplify(simplify_tolerance, preserve_topology=True)

                # handle case where simplification results in MultiPolygon
                if isinstance(poly_simplified, MultiPolygon):
                    poly_simplified = max(poly_simplified.geoms, key=lambda p: p.area)

                # check if we still have a valid polygon
                if hasattr(poly_simplified, 'exterior'):
                    polygon_coords = list(poly_simplified.exterior.coords)
                else:
                    # Simplification failed, keep original
                    print(f"Warning: Simplification resulted in invalid geometry for instance {i}, using original contour")
            except Exception as e:
                print(f"Warning: Simplification failed for instance {i}, using original contour: {e}")

        # validate polygon has enough points before closing
        if len(polygon_coords) < 3:
            continue

        # ensure polygon is closed (first point == last point)
        if polygon_coords[0] != polygon_coords[-1]:
            polygon_coords.append(polygon_coords[0])

        # validate polygon has enough points
        if len(polygon_coords) < 4:  # need at least 3 unique points + closing point
            continue

        # create GeoJSON polygon
        # geoJSON polygon format: coordinates are [[[x, y], [x, y], ...]]
        polygon = Polygon([polygon_coords])

        feature = Feature(
            geometry=polygon,
            properties={
                "id": int(i),
                "mask_value": float(value),
                "area": float(instance_mask.sum()),
                "perimeter": float(cv2.arcLength(contour, True))
            }
        )
        features.append(feature)

    feature_collection = FeatureCollection(features)

    with open(output_file, "w") as f:
        json.dump(feature_collection, f, indent=2)

    print(f"GeoJSON with {len(features)} polygon instances saved to {output_file}")


def main():
    ## Example 1: YOLO labels to GeoJSON
    # label_file = "/Users/simon/Downloads/kaggle 3/working/yolo_dataset/labels/train/tile_0013_640_1280.txt"
    # image_width = 640
    # image_height = 640
    # yolo_to_geojson(label_file, image_width, image_height, output_file="/Users/simon/Downloads/tile_0013_640_1280.geojson")
    
    ## Example 2: Mask to GeoJSON with bounding boxes
    # mask_file = '/Users/simon/Downloads/stitched_mask (1).npy'
    # npy_mask_to_geojson_bbox(mask_file, output_file='/Users/simon/Downloads/stitched_mask_bbox.geojson')
    
    ## Example 3: Mask to GeoJSON with precise polygons
    mask_file = '/Users/simon/Downloads/stitched_mask (1).npy'
    npy_mask_to_geojson_polygon(
        mask_file, 
        output_file='/Users/simon/Downloads/stitched_mask_polygon.geojson',
        simplify_tolerance=1.0  # adjust this to balance precision vs file size
    )


if __name__ == "__main__":
    main()
