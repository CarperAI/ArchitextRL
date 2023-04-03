import os
import random
import numpy as np
from abc import ABC
import math
import re
from omegaconf import DictConfig, OmegaConf
from shapely.geometry.polygon import Polygon
from shapely.geometry import shape
from shapely.affinity import scale
from shapely.ops import unary_union
import networkx as nx
from typing import List
from PIL import Image, ImageDraw


def draw_polygons(polygons, colors, im_size=(256, 256), b_color="white", fpath=None):
    image = Image.new("RGBA", im_size, color="white")  # Image.new("L", im_size, color="white")
    draw = ImageDraw.Draw(image)

    for poly, color, in zip(polygons, colors):
        xy = poly.exterior.xy
        coords = np.dstack((xy[1], xy[0])).flatten()
        draw.polygon(list(coords), fill=(0, 0, 0))

        # get inner polygon coordinates
        small_poly = poly.buffer(-1, resolution=32, cap_style=2, join_style=2, mitre_limit=5.0)
        if small_poly.geom_type == 'MultiPolygon':
            mycoordslist = [list(x.exterior.coords) for x in small_poly]
            for coord in mycoordslist:
                coords = np.dstack((np.array(coord)[:, 1], np.array(coord)[:, 0])).flatten()
                draw.polygon(list(coords), fill=tuple(color))
        elif small_poly.geom_type == 'Polygon':
            # get inner polygon coordinates
            if not small_poly.is_empty:
                xy2 = small_poly.exterior.xy
                coords2 = np.dstack((xy2[1], xy2[0])).flatten()
                # draw it on canvas, with the appropriate colors
                draw.polygon(list(coords2), fill=tuple(color))

                # image = image.transpose(Image.FLIP_TOP_BOTTOM)

    if (fpath):
        image.save(fpath, format='png', quality=100, subsampling=0)
        np.save(fpath, np.array(image))

    return draw, image


def calc_entropy(labels, base=None):
    """ Computes entropy of label distribution. """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    # Compute entropy
    base = math.e if base is None else base
    ent = -sum(i * math.log(i, base) for i in probs)

    return ent


def get_value(dictionary, val):
    for key, value in dictionary.items():
        if val == key:
            return value

    return "value doesn't exist"


def get_key(dictionary, val):
    for key, value in dictionary.items():
        if val == value:
            return key

    return "key doesn't exist"


def find_intersections(seed_polygon, target_polygons):
    """
        A function that finds intersections between a seed polygon and a list of candidate polygons.

    Args:
        seed_polygon (shapely polygon): A shapely polygon.
        target_polygons (list): A list of shapely polygons.

    Returns:
        array: The intersection matrix between the seed polygon and all individual target polygons.
    """
    intersect_booleans = []
    for _, poly in enumerate(target_polygons):
        try:
            intersect_booleans.append(seed_polygon.intersects(poly))
        except:
            intersect_booleans.append(True)
    return intersect_booleans


def find_distance(seed_graph, target_graphs):
    """
        A function that finds intersections between a seed polygon and a list of candidate polygons.

    Args:
        seed_polygon (shapely polygon): A shapely polygon.
        target_polygons (list): A list of shapely polygons.

    Returns:
        array: The intersection matrix between the seed polygon and all individual target polygons.
    """
    distances = [nx.graph_edit_distance(seed_graph, graph) for graph in target_graphs]
    return distances


housegan_labels = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "missing": 5, "closet": 6,
                   "balcony": 7, "corridor": 8, "dining_room": 9, "laundry_room": 10}
regex = re.compile(r".*?\((.*?)\)")


