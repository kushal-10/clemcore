import os 
import json

# registry = os.path.join("backends", "model_registry.json")

# with open(registry, "r") as f:
#     registry = json.load(f)

# blank_registry = []

# for model in registry:
#     # Add new fields
#     model_name = model["model_name"]
#     model_name = model_name.lower()
#     if "qwen" in model_name or "intern" in model_name:
#         model["languages"] = ["zh", "en"]
#     else:
#         model["languages"] = ["en"]
#     model["context_size"] = ""
#     model["license"] = {
#         "name": "Apache 2.0",
#         "url": "https://www.apache.org/licenses/LICENSE-2.0"
#     }

#     # Update old fields
#     if "estimated_parameters" in model:
#         if model['parameters'] != "":
#             model['parameters'] = model['estimated_parameters']
#             model['parameters_estimated'] = True
#             model.pop("estimated_parameters")


#     if "support_multiple_images" in model:
#         model.pop("support_multiple_images")
#         model["multimodality"] = {
#             "single_image": True,
#             "multiple_images": True,
#             "audio": False,
#             "video": False
#         }

#     if "supports_multiple_images" in model:
#         model.pop("supports_multiple_images")
#         model["multimodality"] = {
#             "single_image": True,
#             "multiple_images": True,
#             "audio": False,
#             "video": False
#         }

#     if "supports_images" in model:
#         model.pop("supports_images")

#     if 'model_id' in model:
#         model_key = 'model_id'
#     elif 'huggingface_id' in model:
#         model_key = 'huggingface_id'
#     else:
#         print("EPIC FAIL!!")
    
#     main_fields = ["model_name", "backend", model_key,  "release_date", "open_weight", "parameters", "languages", "context_size", "license"]

#     # if model["backend"] == "huggingface_multimodal":
#     registry_keys = list(model.keys())
#     model_config = {k: v for k, v in model.items() if k not in main_fields}
    
#     keys_to_remove = []
#     for k in registry_keys:
#         if k not in main_fields:
#             keys_to_remove.append(k)
#     for k in keys_to_remove:
#         model.pop(k)
#     model["model_config"] = model_config

#     blank_registry.append(model)

# with open(os.path.join("backends", "model_registry_new.json"), "w") as f:
#     json.dump(blank_registry, f, indent=4)
    

## Update the data from previous updated registry


with open('backends/model_registry_new.json', 'r') as f:
    new_data = json.load(f)

with open('backends/model_registry_updated.json', 'r') as f:
    updated_data = json.load(f)

with open('backends/model_registry.json', 'r') as f:
    main_data = json.load(f)

final_data = []
for i in range(len(new_data)):
    new_obj = new_data[i]
    for j in range(len(updated_data)):
        updated_obj = updated_data[j]

        if 'model_id' in new_obj:
            model_key = 'model_id'
        else:
            model_key = 'huggingface_id'
        
        if model_key in new_obj and model_key in updated_obj:
            if updated_obj[model_key] == new_obj[model_key]:
                final_obj = new_obj
                final_obj['languages'] = updated_obj['languages']
                final_obj['license'] = updated_obj['license']
                final_obj['context_size'] = updated_obj['context_size']
                final_data.append(final_obj)
                break

    final_obj = new_obj
    final_data.append(final_obj)

with open('backends/model_registry_final.json', 'w') as f:
    json.dump(final_data, f, indent=4)

