import json 

with open('clemcore/backends/model_registry.json', 'r') as f:
    json_data = json.load(f)

print(len(json_data))
