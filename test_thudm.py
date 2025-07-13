from transformers import AutoProcessor, Glm4vForConditionalGeneration
import torch

MODEL_PATH = "THUDM/GLM-4.1V-9B-Thinking"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png"
            },
            {
                "type": "text",
                "text": "describe this image"
            }
        ],
    }
]
clem_messages = [
    {"role": "user", "content": "describe this image", "image":["https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png"]},
    # {"role": "assistant", "content": "some random response"},
    # {"role": "user", "content": "some random text 2"},
    # {"role": "assistant", "content": "some random response 2"},
    # {"role": "user", "content": "some random text 3", "image":["url2"]},
]

def get_glm_messages(clem_messages):
    glm_messages = []
    for msg in clem_messages:
        msg_dict = {}
        if msg["role"] == "user":
            msg_dict["role"] = "user"
            msg_dict["content"] = [
                {
                  "type": "text",
                  "content": msg["content"],
                }
            ]
            if "image" in msg:
                for img in msg["image"]:
                    msg_dict["content"].append({"type": "image", "content": img})
        elif msg["role"] == "assistant":
            msg_dict["role"] = "assistant"
            msg_dict["content"] = [
                {
                    "type": "text",
                    "content": msg["content"],
                }
            ]

        glm_messages.append(msg_dict)
    return glm_messages



glm_messages = get_glm_messages(clem_messages)
print(glm_messages)

processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
# model = Glm4vForConditionalGeneration.from_pretrained(
#     pretrained_model_name_or_path=MODEL_PATH,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
inputs = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

print(inputs)
# generated_ids = model.generate(**inputs, max_new_tokens=8192)
# output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
# print(output_text)
