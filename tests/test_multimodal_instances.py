import json

# game_name = "textmapworld_description"

game_names = [
    "referencegame"
]

def gen_test_instances(game_name:str):
    with open(f'games/{game_name}/in/instances.json', 'r') as f:
        json_data = json.load(f)

    pentomino_data = None
    if "matchit" in game_name and "ascii" not in game_name:
        splits = game_name.split("_")
        if len(splits) != 1:
            with open(f'games/{game_name}/in/instances_{splits[-1]}_pentomino.json', 'r') as f:
                pentomino_data = json.load(f)
        else:
            with open(f'games/{game_name}/in/instances_base_pentomino.json', 'r') as f:
                pentomino_data = json.load(f)
            

    test_data = {
        "experiments": [

        ]
    }

    exps = json_data['experiments']
    for e in exps:
        dict_obj = {
            "name": "",
            "game_instances": [

            ]
        }

        insts = e["game_instances"]
        dict_obj["name"] = e["name"]
        for i in insts:
            if i["game_id"] == 0: # mm_ref
            # if i["game_id"] %10 == 0: # textmap
                dict_obj["game_instances"].append(i)

        test_data["experiments"].append(dict_obj)

    if pentomino_data:
        exps = pentomino_data['experiments']
        for e in exps:
            dict_obj = {
                "name": "",
                "game_instances": [

                ]
            }

            insts = e["game_instances"]
            dict_obj["name"] = e["name"]
            for i in insts:
                # if i["game_id"] == 0: # mm_ref
                if i["game_id"] %10 == 0: # textmap
                    dict_obj["game_instances"].append(i)

            test_data["experiments"].append(dict_obj)


    with open(f'games/{game_name}/in/test.json', 'w') as f:
        json.dump(test_data, f, indent=4)

    
for g in game_names:
    gen_test_instances(g)