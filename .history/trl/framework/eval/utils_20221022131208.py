from functools import reduce
import re
import os
import time
from typing import Iterable, List, Any, Callable
import pickle
import numpy as np

#Architext specific dependencies
from shapely.geometry.polygon import Polygon
from shapely.affinity import scale
from math import atan2
import networkx as nx
import num2word
from word2number import w2n

housegan_labels = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "missing": 5, "closet": 6, 
                         "balcony": 7, "corridor": 8, "dining_room": 9, "laundry_room": 10}

# Generic utilities 

regex = re.compile(".*?\((.*?)\)")

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

def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

#Geometry specific utilities
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


def extract_layout_properties(layout):
    if(len(layout.split('[layout]')) > 1):
        layout = layout.split('[layout]')[1].split('[User prompt]')[0].split(', ')
    else:
        layout = layout.split('[Layout]')[1].split('[User prompt]')[0].split(', ')
    spaces = [re.sub(r'\d+', '', txt.split(':')[0]).lstrip() for txt in layout]
    space_ids = [get_value(housegan_labels, space) for space in spaces]
    coordinates = [txt.split(':')[1] for txt in layout if len(txt.split(':')) > 1]
    coordinates = [re.findall(regex, coord) for coord in coordinates]
    coordinates = [x for x in coordinates if x != []]
    polygons = []
    for coord in coordinates:
        polygons.append([point.split(',') for point in coord])

    return spaces, space_ids, polygons

# Graph metrics / evaluations
def adjacency_matrix(space_ids, polygons):
    scaled_polygons = []
    for polygon in polygons:
        scaled_polygons.append(scale(polygon, 1.15, 1.15, origin=polygon.centroid))
    intersection_matrix = np.zeros((len(scaled_polygons), len(scaled_polygons)))
    for k, p in enumerate(scaled_polygons):
        intersection_matrix[:, k] = find_intersections(p, scaled_polygons)
    return intersection_matrix

def create_node_dict(space_ids):
    values = [get_key(housegan_labels, id_) for id_ in space_ids]
    keys = np.arange(len(space_ids))
    return dict(zip(keys, values))

def store_as_list_of_dicts(filename, *graphs):

    list_of_dicts = [nx.to_dict_of_dicts(graph) for graph in graphs]

    with open(filename, 'wb') as f:
        pickle.dump(list_of_dicts, f)
    
def load_list_of_dicts(filename, create_using=nx.Graph):
    
    with open(filename, 'rb') as f:
        list_of_dicts = pickle.load(f)
        
    graphs = [create_using(graph) for graph in list_of_dicts]
    
    return graphs

def find_distance(seed_graph, target_graphs):
    """
        A function that calculates graph edit distances between a seed layout and a list of target layouts.

    Args:
        seed_graph (nx graph): A graph representation of the layout
        target_graphs (list): A list of graphs.

    Returns:
        list: the pairwise distances for the seed graph
    """
    distances = [nx.graph_edit_distance(seed_graph, graph) for graph in target_graphs]
    return distances

# Room specific utilities
room_angles = np.array([[0, 22.5], 
                     [22.5, 67.5], 
                     [67.5, 112.5], 
                     [112.5, 157.5], 
                     [157.5, 180], 
                     [-157.5, -180],
                     [-112.5, -157.5],                    
                     [-67.5, -112.5],
                     [-22.5, -67.5],
                     [0, -22.5]], dtype=float)

room_orientations = np.array(['north', 
                              'north east', 
                              'east', 
                              'south east', 
                              'south', 
                              'south',       
                              'south west', 
                              'west', 
                              'north west', 
                              'north'], dtype=str)

def angle_between(a,b):
    dot = a[0]*b[0] + a[1]*b[1]      # dot product between [x1, y1] and [x2, y2]
    det = a[0]*b[1] - a[1]*b[0]      # determinant
    angle = atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    
    return np.rad2deg(angle)

location_adjacencies = {'north': ['north east', 'north west'],
                        'north west': ['north', 'west'],
                        'west': ['north west', 'south west'],
                        'south west': ['west', 'south'],
                        'south': ['south west', 'south east'],
                        'south east': ['south', 'east'],
                        'east': ['south east', 'north east'],
                        'north east': ['north', 'east']}

# Layout specific utilities
def house_bbox(polygons):
    bounds = [polygon.bounds for polygon in polygons]
    
    xmin = np.array(bounds)[:, 0].min()
    ymin = np.array(bounds)[:, 1].min()
    xmax = np.array(bounds)[:, 2].max()
    ymax = np.array(bounds)[:, 3].max()
    
    return xmin, xmax, ymin, ymax

def get_room_centroids(geometry):

    room_coords = np.concatenate([poly.centroid.xy for poly in geometry])
    room_y = room_coords[0::2]
    room_x = room_coords[1::2]
    room_centroids = np.hstack([room_x, room_y])
    return room_centroids

def get_room_vectors(geom, room_centroids):
    vectors = []
    xmin, xmax, ymin, ymax = house_bbox(geom)
    center_point = xmin+((xmax - xmin)/2), ymin+((ymax-ymin)/2)
    for centroid in room_centroids:
        vectors.append([center_point[0]-centroid[0], center_point[1]-centroid[1]])
    return vectors

def num_rooms_annotation(spaces):
    
    desc = []
    nbed = np.where(np.array(spaces) == 'bedroom')[0].shape[0]
    nbath = np.where(np.array(spaces) == 'bathroom')[0].shape[0]
    if((nbed > 1) & ((nbath) > 1)):
        desc.append("a house with %s bedrooms and %s bathrooms" % (num2word.word(nbed).lower(), 
                                                                       num2word.word(nbath).lower()))
    elif((nbed > 1) & ((nbath) == 1)):
        desc.append("a house with %s bedrooms and %s bathroom" % (num2word.word(nbed).lower(), 
                                                                      num2word.word(nbath).lower()))
    elif((nbed == 1) & ((nbath) > 1)):
        desc.append("a house with %s bedroom and %s bathrooms" % (num2word.word(nbed).lower(), 
                                                                      num2word.word(nbath).lower()))
    elif((nbed == 1) & ((nbath) == 1)):
        desc.append("a house with %s bedroom and %s bathroom" % (num2word.word(nbed).lower(), 
                                                                     num2word.word(nbath).lower()))
    else:
        if(nbath == 0):
            if(nbed > 1):
                desc.append("a house with %s bedrooms and no bathroom" % (num2word.word(nbed).lower()))
            elif(nbed==0):
                desc.append("a house with no bedroom and no bathroom")
            else:
                desc.append("a house with %s bedroom and no bathroom" % (num2word.word(nbed).lower()))
        elif(nbed == 0):
            if(nbath > 1):
                desc.append("a house with no bedroom and %s bathrooms" % (num2word.word(nbath).lower()))
            elif(nbath == 0):
                desc.append("a house with no bedroom and no bathrooms")
            else:
                desc.append("a house with no bedroom and %s bathroom" % (num2word.word(nbath).lower()))
        else:
            desc.append("a house with %s bedrooms and %s bathrooms" % (num2word.word(nbed).lower(), 
                                                                           num2word.word(nbath).lower()))
    if(len(spaces)>1):
        if('corridor' in spaces):
            desc.append("a house with %s rooms and a corridor" % num2word.word(len(spaces)-1).lower())
        else:
            desc.append("a house with %s rooms" % num2word.word(len(spaces)).lower())    
    else:
        desc.append("a house with %s room" % num2word.word(len(spaces)).lower())

    #if("corridor" in spaces):
    #    desc[0] = desc[0].replace(' and', ',') + " and a corridor"
    #    desc[1] = desc[1].replace(' and', ',') + " and a corridor"
    #    desc[1] = desc[1].replace(num2word.word(len(spaces)).lower(), num2word.word(len(spaces)-1).lower())
        
    return desc

def location_annotations(spaces, vectors):
    init_vector = [0,-1]
    desc = []
    loc_descriptions = []
    
    kept_spaces = ["living_room", "kitchen", "bedroom", "bathroom", "dining_room", "corridor"]
    nbed = np.where(np.array(spaces) == 'bedroom')[0].shape[0]
    nbath = np.where(np.array(spaces) == 'bathroom')[0].shape[0]
    
    for space, vector in zip(spaces, vectors):
        angle = int(angle_between(vector, init_vector))
        if(angle>0):
            cond = [(angle >= int(orientation[0])) & (angle <= int(orientation[1])) for orientation in room_angles]
        else:
            cond = [(angle <= int(orientation[0])) & (angle >= int(orientation[1])) for orientation in room_angles]
        loc = room_orientations[cond][0]
        if(space in ['kitchen', 'living_room', 'corridor']):
            loc_descriptions.append('the %s is located in the %s side of the house' % (space, loc))
        elif(space=='bathroom'):
            if(nbath==1):
                loc_descriptions.append('the %s is located in the %s side of the house' % (space, loc))
            else:
                loc_descriptions.append('a %s is located in the %s side of the house' % (space, loc))
        elif(space=='bedroom'):
            if(nbed==1):
                loc_descriptions.append('the %s is located in the %s side of the house' % (space, loc))
            else:
                loc_descriptions.append('a %s is located in the %s side of the house' % (space, loc))

    desc.append(list(set(flatten(loc_descriptions))))
    
    return desc