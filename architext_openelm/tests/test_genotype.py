from architext_genotype import ArchitextGenotype


def test_genotype():
    prompt = """[prompt] a bedroom is adjacent to the kitchen [layout] bedroom1: (194,91)(135,91)(135,47)(194,47), living_room: (121,194)(47,194)(47,91)(106,91)(106,106)(121,106), bathroom1: (179,121)(135,121)(135,91)(179,91), bedroom2: (209,165)(135,165)(135,121)(209,121), bathroom2: (165,209)(135,209)(135,165)(165,165), bedroom3: (121,238)(47,238)(47,194)(121,194), kitchen: (135,77)(106,77)(106,18)(135,18), corridor: (121,209)(121,106)(106,106)(106,77)(135,77)(135,209) <|endoftext|>\n"""

    prompt_overlapping = """[prompt] a bedroom is adjacent to the kitchen [layout] bedroom1: (194,91)(130,91)(130,47)(194,47), living_room: (121,194)(47,194)(47,91)(106,91)(106,106)(121,106), bathroom1: (179,121)(135,121)(135,91)(179,91), bedroom2: (209,165)(135,165)(135,121)(209,121), bathroom2: (165,209)(135,209)(135,165)(165,165), bedroom3: (121,238)(47,238)(47,194)(121,194), kitchen: (135,77)(106,77)(106,18)(135,18), corridor: (121,209)(121,106)(106,106)(106,77)(135,77)(135,209) <|endoftext|>\n"""

    prompt_missing = """[prompt] a bedroom is adjacent to the kitchen [layout] bedroom1: (194,91)(135,91)(135,47)(194,47), living_room: (121,194)(47,194)(47,91)(106,91)(106,106)(121,106), bathroom1: (179,121)(135,121)(135,91)(179,91), bedroom2: (209,165)(135,165)(135,121)(209,121), bathroom2: (165,209)(135,209), bedroom3: (121,238)(47,238)(47,194)(121,194), kitchen: (135,77)(106,77)(106,18)(135,18), corridor: (121,209)(121,106)(106,106)(106,77)(135,77)(135,209) <|endoftext|>\n"""

    prompt_disjoint = """[prompt] a bedroom is adjacent to the kitchen [layout] bedroom1: (194,91)(135,91)(135,47)(194,47), living_room: (121,194)(47,194)(47,91)(106,91)(106,106)(121,106), bathroom1: (179,121)(135,121)(135,91)(179,91), bedroom2: (209,165)(135,165)(135,121)(209,121), bathroom2: (165,209)(135,209)(135,165)(165,165), bedroom3: (1210,2380)(470,2380)(470,1940)(1210,1940), kitchen: (135,77)(106,77)(106,18)(135,18), corridor: (121,209)(121,106)(106,106)(106,77)(135,77)(135,209) <|endoftext|>\n"""


    json = {"prompt":
                "a bedroom is adjacent to the kitchen",
            "layout":
                {
                    "bedroom1": [["194", "91"], ["135", "91"], ["135", "47"], ["194", "47"]],
                    "living_room": [["121", "194"], ["47", "194"], ["47", "91"], ["106", "91"], ["106", "106"],
                                    ["121", "106"]],
                    "bathroom1": [["179", "121"], ["135", "121"], ["135", "91"], ["179", "91"]],
                    "bedroom2": [["209", "165"], ["135", "165"], ["135", "121"], ["209", "121"]],
                    "bathroom2": [["165", "209"], ["135", "209"], ["135", "165"], ["165", "165"]],
                    "bedroom3": [["121", "238"], ["47", "238"], ["47", "194"], ["121", "194"]],
                    "kitchen": [["135", "77"], ["106", "77"], ["106", "18"], ["135", "18"]],
                    "corridor": [["121", "209"], ["121", "106"], ["106", "106"], ["106", "77"], ["135", "77"],
                                 ["135", "209"]]
                }
            }

    print(ArchitextGenotype(prompt))
    print(ArchitextGenotype(prompt).to_design_string())
    print(ArchitextGenotype(prompt_overlapping))
    print(ArchitextGenotype(prompt_missing))
    print(ArchitextGenotype(prompt_disjoint))


if __name__ == "__main__":
    test_genotype()
