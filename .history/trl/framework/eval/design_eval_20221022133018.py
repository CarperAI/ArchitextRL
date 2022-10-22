from utils import *

def eval_function(prompts, samples, prompt_types):
    semantic_accuracy = []
    reward = []
    # assuming a batch of layouts sampled from the model
    for prompt, layout, prompt_type in zip(prompts, samples, prompt_types):
        geom = []
        try:
            # get layout geometry
            spaces, _, polygons = extract_layout_properties(layout)
            for poly in polygons:
                poly = [x for x in poly if x != ['']]
                poly = [x for x in poly if '' not in x] 
                geom.append(Polygon(np.array(poly, dtype=int)))

            # get geometric properties: centroids and vectors
            room_centroids = get_room_centroids(geom)
            vectors = get_room_vectors(geom, room_centroids)
            
            # get layout annotations based on number of rooms and location
            desc = []
            num_desc = num_rooms_annotation(spaces)
            desc.extend(list(set(flatten(num_desc))))
            loc_desc = location_annotations(spaces, vectors)
            desc.extend(list(set(flatten(loc_desc))))
            desc = [re.sub('_', ' ', d) for d in desc]

            # calculate semantic accuracy: number of generations that satisfy the prompt
            semantic_accuracy.append(prompt in desc)

            # calculate reward according to type of prompt: difference or distance
            type_reward = get_reward(prompt, spaces, desc, prompt_type)
            reward.append(type_reward)
        except:
            # what type of values should we put when the model fails to create a valid design?
            semantic_accuracy.append(-999)
            reward.append(-999)

    results = {'semantic_accuracy': semantic_accuracy, 'reward': reward}
    #results.append((semantic_accuracy, type_reward))
    return results
