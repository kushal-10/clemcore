import json 

with open('backends/model_registry.json', 'r') as f:
    main_data = json.load(f)

with open('backends/model_registry_final.json', 'r') as f:
    final_data = json.load(f)

new_data = []

for d in main_data:
    if 'model_id' in d:
        name = 'model_id'
    else:
        name = 'huggingface_id'
    
    if not d[name] in new_data:
        new_data.append(d)


with open('backends/model_registry_reduced.json', 'w') as f:
    json.dump(new_data, f, indent=4)

