import os 
import json

registry = os.path.join("backends", "model_registry.json")

with open(registry, "r") as f:
    registry = json.load(f)

blank_registry = []

for model in registry:
    # Add new fields
    model_name = model["model_name"]
    model_name = model_name.lower()
    if "qwen" or "intern" in model_name:
        model["languages"] = ["zh", "en"]
    else:
        model["languages"] = ["en"]
    model["context_size"] = ""
    model["license"] = {
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0"
    }

    # Update old fields
    if "estimated_parameters" in model:
        model.pop("estimated_parameters")

    if "T" in model["parameters"]:
        model["parameters"] = ""

    if "support_multiple_images" in model:
        model.pop("support_multiple_images")
        model["multimodality"] = {
            "single_image": True,
            "multiple_images": True,
            "audio": False,
            "video": False
        }

    if "supports_multiple_images" in model:
        model.pop("supports_multiple_images")
        model["multimodality"] = {
            "single_image": True,
            "multiple_images": True,
            "audio": False,
            "video": False
        }

    if "supports_images" in model:
        model.pop("supports_images")

    main_fields = ["model_name", "backend", "huggingface_id",  "release_date", "open_weight", "parameters", "languages", "context_size", "license"]

    if model["backend"] == "huggingface_multimodal":
        registry_keys = list(model.keys())
        model_config = {k: v for k, v in model.items() if k not in main_fields}
        
        keys_to_remove = []
        for k in registry_keys:
            if k not in main_fields:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            model.pop(k)
        model["model_config"] = model_config

    blank_registry.append(model)

with open(os.path.join("backends", "model_registry_updated.json"), "w") as f:
    json.dump(blank_registry, f, indent=4)
    

