import json
import numpy as np
import cv2  # OpenCV for drawing polygons
from shapely.wkt import loads as wkt_loads # For parsing WKT strings

# Mapping from the JSON subtype to the integer value for the mask
DAMAGE_TO_INT_MAP = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "no-damage Building": 1 # Treat un-classified as no-damage for simplicity
}

def generate_mask_from_json(json_path, height=1024, width=1024):
    """
    Reads a JSON label file, parses the building polygons, and generates a
    segmentation mask image.

    Args:
        json_path (str): Path to the xView2 JSON label file.
        height (int): Height of the mask to be generated.
        width (int): Width of the mask to be generated.

    Returns:
        numpy.ndarray: A NumPy array representing the segmentation mask.
    """
    # Start with an empty mask (all zeros - background)
    mask = np.zeros((height, width), dtype=np.uint8)

    with open(json_path, 'r') as f:
        label_data = json.load(f)

    # The polygons are in the 'xy' features
    polygons = label_data['features']['xy']

    for poly in polygons:
        damage_subtype = poly['properties']['subtype']
        
        # Get the integer value for the damage level
        mask_value = DAMAGE_TO_INT_MAP.get(damage_subtype, 0)
        
        # Parse the WKT string to get polygon coordinates
        wkt_string = poly['wkt']
        polygon_shape = wkt_loads(wkt_string)
        
        # Convert shapely polygon to a list of integer coordinates for OpenCV
        exterior_coords = np.array(polygon_shape.exterior.coords).round().astype(np.int32)
        
        # Draw the filled polygon on our empty mask
        cv2.fillPoly(mask, [exterior_coords], mask_value)

    return mask