from utils import *
import re

regex = re.compile(".*?\((.*?)\)")


def eval_function(prompts, samples, prompt_types):
    #results = []
    semantic_accuracy = []
    type_reward = []
    # assuming a batch of generations sampled from the model
    for layout, prompt, prompt_type in zip(prompts, samples, prompt_types):
        try:
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
            geom = []
            new_spaces = []
            new_space_ids = []
            for poly, space, space_id in zip(polygons, spaces, space_ids):
                poly = [x for x in poly if x != ['']]
                poly = [x for x in poly if '' not in x] 
                geom.append(Polygon(np.array(poly, dtype=int)))
                new_spaces.append(space)
                new_space_ids.append(space_id)

            room_centroids = get_room_centroids(geom)
            vectors = get_room_vectors(geom, room_centroids)
            desc = []
            num_desc = num_rooms_annotation(spaces)
            desc.extend(list(set(flatten(num_desc))))
            loc_desc = location_annotations(spaces, vectors)
            desc.extend(list(set(flatten(loc_desc))))
            desc = [re.sub('_', ' ', d) for d in desc]
            semantic_accuracy.append(prompt in desc)

            if(prompt_type == 'ind_number_prompt'):
                req_bed = w2n.word_to_num(prompt.split('with ')[1].split(' ')[0])
                gen_bed = np.where(np.array(spaces) == 'bedroom')[0].shape[0]
                bed_difference = abs(req_bed - gen_bed)
                print(req_bed)
                req_bath = w2n.word_to_num(prompt.split('and ')[1].split(' ')[0])
                gen_bath = np.where(np.array(spaces) == 'bathroom')[0].shape[0]
                bath_difference = abs(req_bath - gen_bath)
                type_reward.append(bed_difference + bath_difference)
            elif(prompt_type == 'total_number_prompt'):
                req_rooms = w2n.word_to_num(prompt.split('with ')[1].split(' ')[0])
                if('corridor' in prompt):
                    req_rooms +=1
                gen_rooms = len(spaces)
                room_difference = -abs(req_rooms - gen_rooms)
                print(room_difference)
                type_reward.append(room_difference)
            elif(prompt_type == 'location_prompt'):
                try:
                    req_location = prompt.split('is located in the ')[1].split(' side')[0]
                    gen_location = [d for d in desc if 'side of the house' in d]
                    gen_location = [loc for loc in gen_location if req_location in loc]
                    gen_cell = gen_location[0].split('is located in the ')[1].split(' side')[0]
                    adj_cells = location_adjacencies[req_location]
                    if gen_cell == req_location:
                        type_reward.append(1)
                    elif gen_cell in adj_cells:
                        type_reward.append(-1)
                    else:
                        type_reward.append(-2)
                except:
                    type_reward.append(-999)
            else:
                type_reward.append(None)
        except:
            # what type of values do we put when the model fails to create a valid design?
            semantic_accuracy.append(-999)
            type_reward.append(-999)


    results = {'semantic_accuracy': semantic_accuracy, 'reward': type_reward}
    #results.append((semantic_accuracy, type_reward))
    return results
