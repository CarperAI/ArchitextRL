import re
from typing import Optional

import numpy as np
from shapely import make_valid
from shapely.geometry import Polygon
from shapely.ops import unary_union

from enum import Enum
from util import calc_entropy, draw_polygons
from openelm.environments import Genotype

Phenotype = Optional[np.ndarray]
pattern = re.compile(r"\[prompt\] +(.*) \[layout\] +(.*) +<\|endoftext\|>")
rooms = re.compile(r"([^ ]+): +([^ ]+)")
coords = re.compile(r"\((-?\d+),(-?\d+)\)")


class ErrorType(Enum):
    Valid = 0
    InvalidString = 1
    InvalidPolygon = 2
    OverlappingPolygons = 3
    DisjointPolygons = 4
    OtherError = 99


class ArchitextGenotype(Genotype):
    visualization_dict = {"living_room": [249, 222, 182],
                          "kitchen": [195, 209, 217],
                          "bedroom": [250, 120, 128],
                          "bathroom": [126, 202, 234],
                          "corridor": [132, 151, 246]}
    end_token_str = '<|endoftext|>'

    def __init__(self,
                 design_string: str,
                 height: float = 2.0,
                 parent=None):
        self.design_string = design_string
        self.design_json = None
        self.design_polygons = None
        self.design_colors = None
        self.design_merged_polygon = None

        self.height = height
        self.parse_design()

        self.parent = parent

        # 1B1B, 2B1B, 2B2B, 3B1B, 3B2B, 3B3B, 4B1B, 4B2B, 4B3B, 4B4B, numbered consecutively
        # label -> typology
        self.typologies_to = {
            num: typ
            for num, typ in enumerate(
                [
                    (i + 1, j + 1) for j in range(4) for i in range(4)
                    if j <= i
                ]
            )}
        # typology -> label
        self.typologies_from = {typ: num for num, typ in self.typologies_to.items()}

    @classmethod
    def from_dict(cls, design_dict: dict, *args, **kwargs):
        design_string = cls._to_design_string(design_dict)
        return cls(design_string, *args, **kwargs)

    def to_phenotype(self) -> Phenotype:
        if not self.valid:
            return None
        else:
            gfa_entropy = self.design_json["metrics"]["gfa_entropy"]
            if self.typology() in self.typologies_from:
                typ = self.typologies_from[self.typology()]
            else:
                return None
            return np.array([gfa_entropy, typ])

    def parse_design(self):
        """
        Parse the design string into a collection of objects for further processing.
          - self.design_json: a json document including a `valid` field and various numeric metrics
          - self.design_polygons: a list of shapely polygons
          - self.design_colors: a list of colors accompanying the polygons
          - self.design_merged_polygon: a merged polygon
        """
        match = pattern.match(self.design_string)
        self.design_json = {"prompt": "", "layout": {}, "metrics": {}, "valid": False}
        self.design_polygons = []
        self.design_colors = []

        if not match or len(match.groups()) < 2:
            return

        self.design_json["prompt"] = match.group(1)
        layout = match.group(2)

        valid = True
        color_regex = re.compile(rf"(" + r"|".join(list(self.visualization_dict.keys())) + rf")")
        for room in rooms.findall(layout):
            self.design_json["layout"][room[0]] = [(x, y) for x, y in coords.findall(room[1])]
            try:
                polygon = make_valid(Polygon(self.design_json["layout"][room[0]]))
                if polygon.geom_type == "MultiPolygon":
                    polygon = list(polygon.geoms)
                elif polygon.geom_type == "Polygon":
                    polygon = [polygon]
                else:
                    raise ValueError("Invalid polygon")

                self.design_polygons.extend(polygon)
                color_string = color_regex.match(room[0]).group(0)
                self.design_colors.extend([self.visualization_dict[color_string]] * len(polygon))
            except ValueError as e:
                self.design_json["error"] = ErrorType.InvalidPolygon
                valid = False
            except Exception as e:
                self.design_json["error"] = ErrorType.OtherError
                valid = False

        self.design_json["valid"] = valid and self.validate()

        if valid:
            try:
                self.design_merged_polygon = unary_union(self.design_polygons)
                # Can be refactored into a loop if we have more metrics
                self.design_json["metrics"]["hlff"] = self.hlff()
                self.design_json["metrics"]["gfa"] = self.gfa()
                self.design_json["metrics"]["gfa_entropy"] = self.gfa_entropy()

                # If there are isolated rooms, the above numbers can still be computed but need to be marked as invalid.
                if self.design_merged_polygon.geom_type == "MultiPolygon" and len(self.design_merged_polygon.geoms) > 1:
                    self.design_json["valid"] = False
                    self.design_json["error"] = ErrorType.DisjointPolygons
            except:
                self.design_json["valid"] = False

    def validate(self) -> bool:
        if not self.design_polygons:
            self.design_json["error"] = ErrorType.InvalidString
            return False
        # Make sure there is no overlap between polygons
        for i in range(len(self.design_polygons)):
            if not self.design_polygons[i].is_valid:
                self.design_json["error"] = ErrorType.InvalidPolygon
                return False
            for j in range(i + 1, len(self.design_polygons)):
                try:
                    if self.design_polygons[i].overlaps(self.design_polygons[j]):
                        self.design_json["error"] = ErrorType.OverlappingPolygons
                        return False
                except:
                    self.design_json["error"] = ErrorType.OtherError
                    return False
        return True

    @staticmethod
    def _to_design_string(design_dict):
        prefix = f"[prompt] {design_dict['prompt']} [layout] "
        coord_strings = []
        for rm in design_dict["layout"]:
            coord = "".join([f"({x},{y})" for x, y in design_dict["layout"][rm]])
            coord_strings.append(f"{rm}: {coord},")
        return prefix + " ".join(coord_strings) + " <|endoftext|>"

    def to_design_string(self) -> str:
        return self._to_design_string(self.design_json)

    def to_dict(self, prompt_layout_only=True) -> dict:
        if prompt_layout_only:
            return {"prompt": self.design_json["prompt"], "layout": self.design_json["layout"]}
        return self.design_json

    # ---- metrics ----
    def hlff(self) -> float:
        # Quality - hlff
        if self.valid:
            surface_area = self.design_merged_polygon.length * self.height
            floor_area = self.design_merged_polygon.area
            hlff = (2 * floor_area + surface_area) / floor_area
            return -hlff
        else:
            return -float("inf")

    def gfa(self) -> float:
        if self.valid:
            polygons = self.design_polygons
            gfa = np.array([poly.area / 14.2 for poly in polygons]).sum()
            return gfa
        else:
            return float("nan")

    def gfa_entropy(self) -> float:
        if self.valid:
            room_gfa = [rm.area for rm in self.design_polygons]
            gfa_entropy = calc_entropy(room_gfa)
            return gfa_entropy
        else:
            return float("nan")

    def typology(self) -> tuple:
        if self.valid:
            rooms = self.design_json["layout"].keys()
            # typologies: [1b1b, 2b1b, 2b2b, 3b1b, 3b2b, 3b3b, 4b1b, 4b2b, 4b3b, 4b4b]
            n_bed = sum(rm.startswith("bedroom") for rm in rooms)
            n_bath = sum(rm.startswith("bathroom") for rm in rooms)
            return n_bed, n_bath
        else:
            return -1, -1

    def get_image(self):
        polygons = self.design_polygons
        colors = self.design_colors
        return draw_polygons(polygons, colors)[1]

    @property
    def valid(self):
        if self.design_json is None or "valid" not in self.design_json or not self.design_json["valid"]:
            return False
        return True

    def _repr_png_(self):
        return self.get_image().tobytes()

    def __str__(self) -> str:
        return str(self.design_json)

    # ---- below are deprecated codes for other use cases ----
    """
    def adjacency_matrix(self):
        scaled_polygons = []
        for polygon in self.design_polygons:
            scaled_polygons.append(scale(polygon, 1.15, 1.15, origin=polygon.centroid))
        intersection_matrix = np.zeros((len(scaled_polygons), len(scaled_polygons)))
        for k, p in enumerate(scaled_polygons):
            intersection_matrix[:, k] = find_intersections(p, scaled_polygons)
        return intersection_matrix

    def create_node_dict(self):
        space_ids = self.get_space_ids()
        values = [get_key(housegan_labels, id_) for id_ in space_ids]
        keys = np.arange(len(space_ids))
        return dict(zip(keys, values))

    def get_labelled_graph(self) -> list:
        adj_matrix = self.adjacency_matrix()
        labels = self.create_node_dict()
        graph = nx.from_numpy_matrix(adj_matrix)
        nx.relabel.relabel_nodes(graph, labels, copy=False)
        return graph
    """
