# pip install accelerate

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch

model_id = "google/gemma-3-27b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

messages = [
    {
        'role': 'user', 
        'content': [
            {'type': 'text', 'text': 'Please help me with the following task. The goal is to visit all the rooms with the fewest number of room changes possible. In each room, you need to decide the direction to go in. Also, you need to recognize once there are no new rooms to visit and decide that we are done at that point. Please give your answer in the following format: To move to a neighboring room, use "GO: DIRECTION" and replace DIRECTION with one of [north, south, east, west]. To stop the exploration, answer with "DONE" instead. Omit any other text.\nHere is an example:\nYou are in the Kitchen. Currently available directions: south, west. What is your next command?\nGO: west\nYou have made a step and entered a Lobby. Currently available directions: east, north. What is your next command?\nGO: north\n...\nYou have made a step and entered a Bedroom. Currently available directions: south. What is your next command?\nDONE\nLet us start. You are in the Cellar. Currently available directions: south, west, east. What is your next command?'}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0]

print(generation)
decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)

# **Overall Impression:** The image is a close-up shot of a vibrant garden scene, 
# focusing on a cluster of pink cosmos flowers and a busy bumblebee. 
# It has a slightly soft, natural feel, likely captured in daylight.
