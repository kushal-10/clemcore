import json

with open('backends/model_registry.json', 'r') as f:
    json_data = json.load(f)

count = 0

for obj in json_data:
    if obj['context_size']:
        count += 1

print(count/len(json_data)*100)
print(count, len(json_data))