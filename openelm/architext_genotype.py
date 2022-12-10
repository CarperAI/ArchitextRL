import re
from typing import Optional

import networkx as nx
import numpy as np
from shapely.affinity import scale
from shapely.geometry import Polygon
from shapely.ops import unary_union

from util import get_value, find_intersections, get_key, calc_entropy, draw_polygons, housegan_labels, regex


# TODO: remove when OpenELM exposes Genotype
class Genotype:
    pass


class ArchitextGenotype(Genotype):

    visualization_dict = {"living_room": [249, 222, 182],
                          "kitchen": [195, 209, 217],
                          "bedroom": [250, 120, 128],
                          "bathroom": [126, 202, 234],
                          "corridor": [132, 151, 246]}
    end_token_str = '<|endoftext|>'

    def __init__(self, code: str,
                 height: float,
                 layout: Optional[str],
                 parent=None):
        self.code = code

        end_index = layout.find(self.end_token_str)
        cut_off_index = end_index + len(self.end_token_str) if end_index != -1 else None
        self.layout = layout[:cut_off_index].strip()

        self.height = height
        self.valid = self.validate()

        self.parent = parent

    def get_clean_layout(self) -> list[str]:
        if len(self.layout.split('[layout]')) > 1:
            clean_layout = self.layout.split('[layout]')[1].split('[prompt]')[0].split(', ')
        else:
            clean_layout = self.layout.split('[Layout]')[1].split('[prompt]')[0].split(', ')
        return clean_layout

    def get_spaces(self) -> list:
        clean_layout = self.get_clean_layout()
        spaces = [re.sub(r'\d+', '', txt.split(':')[0]).lstrip() for txt in clean_layout]
        return spaces

    def get_space_ids(self) -> list:
        spaces = self.get_spaces()
        space_ids = [get_value(housegan_labels, space) for space in spaces]
        return space_ids

    def get_coordinates(self) -> list:
        clean_layout = self.get_clean_layout()
        coordinates = [txt.split(':')[1] for txt in clean_layout if len(txt.split(':')) > 1]
        coordinates = [re.findall(regex, coord) for coord in coordinates]
        coordinates = [x for x in coordinates if x != []]
        return coordinates

    def get_polygons(self) -> list:
        coordinates = self.get_coordinates()
        rectangles = []
        polygons = []
        for coord in coordinates:
            rectangles.append([point.split(',') for point in coord])
        for rec in rectangles:
            rec = [x for x in rec if x != ['']]
            rec = [x for x in rec if '' not in x]
            polygons.append(Polygon(np.array(rec, dtype=int)))

        return polygons

    def get_colors(self) -> list:
        spaces = self.get_spaces()
        colors = [self.visualization_dict[space] for space in spaces]
        return colors

    def validate(self) -> bool:
        try:
            # Make sure the hlff and gfa_entropy methods execute without problems,
            # and the results can be cast into float.
            float(self.hlff())
            float(self.gfa_entropy())
            # Make sure the image can be generated without problems
            self.get_image()
        except:
            return False
        return True

    def adjacency_matrix(self):
        scaled_polygons = []
        for polygon in self.get_polygons():
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

    def hlff(self) -> float:
        # Quality - hlff
        joined = unary_union(self.get_polygons())  # need to add this property to individual
        surface_area = joined.length * self.height  #
        floor_area = joined.area
        hlff = (2 * floor_area + surface_area) / floor_area
        return -hlff

    def gfa(self) -> float:
        polygons = self.get_polygons()
        gfa = np.sum(np.array([poly.area/14.2 for poly in polygons]))
        return gfa

    def gfa_entropy(self) -> float:
        room_gfa = [rm.area for rm in self.get_polygons()]
        gfa_entropy = calc_entropy(room_gfa)
        return gfa_entropy

    def typology(self) -> int:
        spaces = self.get_spaces()
        # typologies: [1b1b, 2b1b, 2b2b, 3b1b, 3b2b, 3b3b, 4b1b, 4b2b, 4b3b, 4b4b]
        typologies = [(i, j) for i in range(1, 5) for j in range(1, 4)]
        nbed = np.where(np.array(spaces) == 'bedroom')[0].shape[0]
        nbath = np.where(np.array(spaces) == 'bathroom')[0].shape[0]
        return typologies.index((nbed, nbath))

    def get_image(self):
        polygons = self.get_polygons()
        colors = self.get_colors()
        return draw_polygons(polygons, colors)[1]

    def _repr_png_(self):
        return self.get_image().tobytes()

    def __str__(self) -> str:
        return self.layout if self.valid else ""
