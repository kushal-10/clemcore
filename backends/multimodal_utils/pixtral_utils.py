from vllm import LLM
from vllm.sampling_params import SamplingParams
import os
from huggingface_hub import login
from typing import Dict, List, Any, Union, Tuple
import base64
from transformers import AutoTokenizer

from backends.multimodal_utils.base_utils import BaseMLLM

# Retrieve the token from environment variables
token = os.getenv("HUGGINGFACE_TOKEN")
if token is None:
    raise ValueError("HUGGINGFACE_TOKEN environment variable not set")
# Authenticate with the Hugging Face Hub
login(token=token)

def image_to_base64(image_path):
    if image_path.startswith("http"):
        return image_path
    else:
        # Open the image file in binary mode
        with open(image_path, "rb") as image_file:
            # Read the file and encode it to base64
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            # Create the data URL
            mime_type = "image/jpeg"  # Adjust according to your image format (e.g., image/png)
            data_url = f"data:{mime_type};base64,{encoded_image}"
            return data_url


class PixtralMLLM(BaseMLLM):
    @staticmethod
    def prepare_inputs(messages: List[Dict[str, Any]], **kwargs) -> Dict:
        """
        Prepare the inputs for the model, including the prompt, images, and conversation history.
        Ref model card - https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3

        :param messages: A list of dictionaries, where each dictionary contains:
                         - 'role': The role of the message sender ('user' or 'assistant').
                         - 'content': The text content of the message.
                         - 'image': Optional; a single image URL (str) or a list of image URLs (List[str]).
        :param kwargs: Additional keyword arguments that may be used in the process.
        :return: A dictionary containing:
                 - 'prompt': The final prompt to be used by the model.
                 - 'images': A list of image URLs to be processed.
                 - 'processor_kwargs': A dictionary with 'history' (list of user-assistant message pairs). Passed to
                                       generate_outputs and get_tokens
        """

        input_prompt = []
        for message in messages:
            message_dict = {}
            message_dict['content'] = []

            if message['role'] == 'user':
                message_dict['role'] = 'user'
                if 'image' in message:
                    if isinstance(message['image'], str):
                        message_dict['content'].append({"type": "image_url", "image_url": {"url": image_to_base64(message['image'])}})
                    elif isinstance(message['image'], list):
                        # List of images
                        for img in message['image']:
                            message_dict['content'].append({"type": "image_url", "image_url": {"url": image_to_base64(img)}})
                    else:
                        raise ValueError("Invalid image type in message - should be str or List[str]")

                # Add user text message at the end
                message_dict['content'].append({"type": "text", "text": message['content']})
                input_prompt.append(message_dict)

            elif message['role'] == 'assistant':
                message_dict['role'] = 'assistant'
                message_dict['content'] = message['content']
                input_prompt.append(message_dict)

            elif message['role'] == 'system':
                sys_content = message['content']
                if sys_content:
                    # Add content only if system message is found
                    message_dict['role'] = 'system'
                    message_dict['content'].append({"type": "text", "text": sys_content})

                    input_prompt.append(message_dict)

        return {
            "prompt": input_prompt,
            "images": [],
            "output_kwargs": {"device": kwargs.get('device'), "max_tokens": kwargs.get('max_tokens')}
        }

    @staticmethod
    def get_tokens(prompt: list, handler: AutoTokenizer, **output_kwargs) -> List[str]:
        """
        Generate tokens for the given prompt and conversation history.

        :param prompt: The current prompt to be tokenized.
        :param handler: The tokenizer/processor used for tokenizing the prompt and history.
        :param kwargs: Additional keyword arguments,

        :return: A list of tokens generated from the combined prompt and conversation history.
        """
        # Convert the prompt to Mistral type
        # input_prompt = handler.apply_chat_template(prompt)
        # Tokenize the combined text
        #tokens = handler.tokenize(input_prompt)

        return [0]

    @staticmethod
    def generate_outputs(prompt: list, images: None, model,
                         handler: AutoTokenizer, **output_kwargs) -> Tuple[Dict[str, Any], str]:
        """
        Generate model outputs given a prompt, images, and additional parameters.

        :param prompt: The list of prompt to be used for generating the response.
        :param images: A list of image URLs or paths to be included in the model's input.
        :param model: The model used for generating the output. This should be compatible with Idefics3.
        :param handler: The processor used to preprocess the prompt and handle the input images.
        :param kwargs: Additional keyword arguments for the model, expected to include 'history'.
        :return:
             - response (Dict[str, Any]): The raw output from the model, formatted as a dictionary.
             - response_text (str): The decoded text response generated by the model.
        """

        device = output_kwargs.get("device")
        max_tokens = output_kwargs.get("max_tokens")

        sampling_params = SamplingParams(max_tokens=max_tokens)

        outputs = model.chat(prompt, sampling_params=sampling_params)

        response_text = outputs[0].outputs[0].text
        response_text = response_text.strip()

        # Format response
        response = {"response": response_text}

        return response, response_text
        
