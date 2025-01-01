import json 

with open('backends/model_registry.json', 'r') as f:
    main_data = json.load(f)

with open('backends/model_registry_reduced.json', 'r') as f:
    final_data = json.load(f)


print(len(final_data), len(main_data))

