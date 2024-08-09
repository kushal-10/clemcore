# Inference class for InternVL series models
# Ref - https://huggingface.co/collections/OpenGVLab/internvl-20-667d3961ab5eb12c7ed1463e

import math
import torch
from typing import Dict, List, Tuple, Any
import torchvision.transforms as T
from PIL import Image
from io import BytesIO
import requests
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from backends.multimodal_utils.base_utils import BaseMLLM

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def split_model(model_name):
    """
    :param model_name: The name of the model (type InternVL2) to split.
    :return: device map for the model.
    """
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


class InternvlMLLM(BaseMLLM):

    @staticmethod
    def build_transform(input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    @staticmethod
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        if image_file.startswith("http"):
            image = Image.open(BytesIO(requests.get(image_file).content))
        else:
            image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def prepare_inputs(self, messages, **model_kwargs):
        """
        Prepare the inputs for the model, including the prompt, images, and conversation history.

        :param messages: A list of dictionaries, where each dictionary contains:
                         - 'role': The role of the message sender ('user' or 'assistant').
                         - 'content': The text content of the message.
                         - 'image': Optional; a single image URL (str) or a list of image URLs (List[str]).
        :param model_kwargs: Additional keyword arguments that may be used in the process.
        :return: A dictionary containing:
                 - 'prompt': The final prompt to be used by the model.
                 - 'images': A list of image URLs to be processed.
                 - 'processor_kwargs': Additional kwargs that may be used in the process including model_kwargs.
        """

        images = []
        conversation_history = []

        for message in messages:
            if message['role'] == 'user':
                previous_user_message = message['content']
                if 'image' in message:
                    if isinstance(message['image'], str):
                        # Single image string
                        images.append(message['image'])  # Append the path of the image
                    elif isinstance(message['image'], list):
                        # List of images
                        for img in message['image']:
                            images.append(img)
                    else:
                        raise ValueError("Invalid image type in message - should be str or List[str]")

                previous_user_message = "<image>\n" + previous_user_message

            elif message["role"] == "assistant":
                # Append user and assistant messages in sequence
                conversation_history.append((previous_user_message, message['content']))

        # Take the last available User message that does not have a corresponding Assistant message
        question = previous_user_message
        device = model_kwargs.get("device")

        # multi-image multi-round conversation, combined images
        pixel_values = None
        if len(images) == 1:
            pixel_values = self.load_image(images[0], max_num=12).to(torch.bfloat16).to(device)
        elif len(images) > 1:
            pixel_values = self.load_image(images[0], max_num=12).to(torch.bfloat16).to(device)
            for i in range(1, len(images)):
                pixel_values1 = self.load_image(images[i], max_num=12).to(torch.bfloat16).to(device)
                pixel_values = torch.cat((pixel_values, pixel_values1), dim=0)

        return {
            "prompt": question,
            "images": pixel_values,
            "output_kwargs": {"history": conversation_history, "max_tokens": model_kwargs.get("max_tokens")}
        }

    def get_tokens(self, prompt: str, handler: AutoTokenizer, **output_kwargs):
        """
        Generate tokens for the given prompt and conversation history.

        :param prompt: The current prompt to be tokenized.
        :param handler: The tokenizer used for tokenizing the prompt and history.
        :param output_kwargs: Additional keyword arguments, expecting 'history' which is a list of tuples (user message, assistant response).

        :return: A list of tokens generated from the combined prompt and conversation history.
        """

        # Extract conversation history from kwargs
        history = output_kwargs.get("history")

        # Combine the prompt with the conversation history
        combined_text = prompt + "".join([user_msg + assistant_response for user_msg, assistant_response in history])

        # Tokenize the combined text
        tokens = handler.tokenize(combined_text)

        return tokens

    def generate_outputs(self, prompt: str, images: List[str], model: AutoModel,
                         handler: AutoTokenizer, **output_kwargs) -> Tuple[Dict[str, Any], str]:

        """
         Generate model outputs given a prompt, images, and additional parameters.

        :param prompt: The text prompt to be used for generating the response.
        :param images: A list of image URLs or paths to be included in the model's input.
        :param model: The model used for generating the output. This should be compatible with InternLM type models.
        :param handler: The tokenizer used to preprocess the prompt and handle the input.
        :param output_kwargs: Additional keyword arguments for the model, expected to include 'history'.
        :return:
             - response (Dict[str, Any]): The raw output from the model, formatted as a dictionary.
             - response_text (str): The decoded text response generated by the model
        """

        history = output_kwargs.get("history")
        max_tokens = output_kwargs.get("max_tokens")

        generation_config = dict(max_new_tokens=max_tokens, do_sample=False)
        generated_response, history = model.chat(handler, images, prompt, generation_config,
                                       history=history, return_history=True)

        # Process and clean response text
        response_text = generated_response.strip()

        # Format response
        response = {"response": generated_response}

        return response, response_text
