import requests
from PIL import Image
from io import BytesIO
import re
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import List, Tuple


def clean_thinking_response(response: str):
    # GLM type responses
    # Try to extract between <answer>...</answer>
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        content = match.group(1).strip()
        if content:  # Not empty
            box_match = re.search(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", content, re.DOTALL)
            if box_match:
                return box_match.group(1).strip()
            return content
        # If empty, keep looking

    # Try to extract between <\|begin_of_box\|>...\|end_of_box\|>
    box_match = re.search(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", response, re.DOTALL)
    if box_match:
        return box_match.group(1).strip()

    # If neither found -> Check for Thinking type responses
    responses_splits = response.split("◁/think▷")
    if len(responses_splits) == 2:
        return responses_splits[-1].strip()
    else:
        return response


def extract_thinking_and_summary(text: str, bot: str = "◁think▷", eot: str = "◁/think▷") -> str:
    if bot in text and eot not in text:
        return ""
    if eot in text:
        return text[text.index(bot) + len(bot):text.index(eot)].strip(), text[text.index(eot) + len(eot) :].strip()
    return "", text

OUTPUT_FORMAT = "--------Thinking--------\n{thinking}\n\n--------Summary--------\n{summary}"

url = "https://huggingface.co/spaces/moonshotai/Kimi-VL-A3B-Thinking/resolve/main/images/demo6.jpeg"

model_path = "moonshotai/Kimi-VL-A3B-Thinking-2506"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

image_paths = [url]
images = [Image.open(BytesIO(requests.get(url).content)) for url in image_paths]
clem_messages = [
    {
        "role": "user", "content": "What kind of cat is this? Answer with one sentence.", "image": image_paths
    }
]

def generate_glm_messages(messages: List[str]) -> Tuple[List, List]:
    glm_messages = []
    for msg in messages:
        glm_message = {"role": msg['role']}
        content_list = [{"type": "text", "text": msg['content']}]
        if 'image' in msg:
            if isinstance(msg['image'], str):
                # Single image
                content_list.append({"type": "image", "url": msg['image']})
            elif isinstance(msg['image'], list):
                # List of images
                for img in msg['image']:
                    content_list.append({"type": "image", "url": img})
            else:
                raise ValueError("Invalid image type in message - should be str or List[str]")
        glm_message['content'] = content_list

        glm_messages.append(glm_message)

    return glm_messages


messages = generate_glm_messages(clem_messages)

text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
inputs = processor(images=images, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
response = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(response)

print("\nCleaned Response\n")
temp_str = "<think>Got it, let's analyze the current room. The TARGET ROOM description is a modern reception area with a curved wooden front desk, polished flooring, tall wooden columns, recessed ceiling lights, and contemporary design elements. The current room shown here has lockers or enclosed spaces with chairs, which doesn't match the reception area description. So it's definitely not the TARGET ROOM. The available directions are north and east. Let's check the options. The action should be MOVE with a direction. Since the current room isn't the target, we need to move. Let's pick north or east. Let's go with north.</think><answer><|begin_of_box|>MOVE: north<|end_of_box|></answer>"
print(clean_thinking_response(temp_str))