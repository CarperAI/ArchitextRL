from design_eval import eval_function
from utils import *

test_data = {
    'samples': ['[prompt] a house with seven rooms and a corridor [layout] bedroom1: (194,106)(165,106)(165,47)(194,47), living_room: (179,223)(106,223)(106,121)(165,121)(165,135)(179,135), bathroom1: (165,106)(135,106)(135,77)(165,77), bedroom2: (135,106)(91,106)(91,33)(135,33), bathroom2: (106,165)(77,165)(77,135)(106,135), bedroom3: (91,106)(77,106)(77,121)(47,121)(47,62)(91,62), kitchen: (209,194)(179,194)(179,135)(194,135)(194,121)(209,121), corridor: (194,135)(165,135)(165,121)(106,121)(106,135)(77,135)(77,106)(194,106) <|endoftext|>',
                '[prompt] a bedroom is located in the east side of the house [layout] bathroom1: (135,99)(91,99)(91,69)(135,69), bedroom1: (121,69)(77,69)(77,25)(121,25), living_room: (179,157)(135,157)(135,69)(179,69), kitchen: (135,157)(91,157)(91,99)(135,99), bedroom2: (179,187)(121,187)(121,157)(179,157), bathroom2: (121,187)(91,187)(91,157)(121,157), bedroom3: (165,231)(106,231)(106,187)(165,187), bedroom4: (179,69)(121,69)(121,25)(179,25) <|endoftext|>',
                '[prompt] a house with two bedrooms and one bathroom [layout] bedroom1: (135,135)(91,135)(91,77)(135,77), living_room: (194,135)(135,135)(135,62)(194,62), kitchen: (194,194)(165,194)(165,135)(194,135), bedroom2: (150,165)(106,165)(106,135)(150,135), bathroom: (106,165)(62,165)(62,135)(106,135) <|endoftext|>'],
    'prompts': ['a house with seven rooms and a corridor', 
                'a bedroom is located in the east side of the house',
                'a house with two bedrooms and one bathroom'],
    'prompt_types': ['total_number_prompt', 'location_prompt', 'ind_number_prompt']
    }


semantic_accuracy = []
reward = []
samples, prompts, prompt_types = test_data['samples'], test_data['prompts'], test_data['prompt_types']
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
        print('fail')

#results = eval_function(test_data['samples'], test_data['prompts'], test_data['prompt_types'])

#print(results)


