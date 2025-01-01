import json 

with open('backends/model_registry.json', 'r') as f:
    main_data = json.load(f)

new_data = []
for i in range(len(main_data)):
    if main_data[i]['backend'] != "huggingface_multimodal":
        new_data.append(main_data[i])
    else:
        obj = main_data[i]
        model_config = obj['model_config']

        if 'model_config' in model_config:
            automodel_config = model_config['model_config']
            model_config.pop('model_config')
            model_config['automodel_config'] = automodel_config
            obj['model_config'] = model_config

            new_data.append(obj)
        else:
            print(model_config)

with open('backends/model_registry.json', 'w') as f:
    json.dump(new_data, f, indent=4)
