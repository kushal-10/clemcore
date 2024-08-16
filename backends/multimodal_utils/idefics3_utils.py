# Inference class for Idefics3-Llama-8b
# Requires installation of transformers from source
# Until this PR is merged - https://github.com/huggingface/transformers/pull/32473
# Working Commit SHA - 0576e9c01e79f6c64411c66420e7810619c35b77 - Use this specifically, others don't work

from typing import Dict, List, Any, Union, Tuple
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

from backends.multimodal_utils.base_utils import BaseMLLM

class Idefics3MLLM(BaseMLLM):

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
        image_paths = []
        for message in messages:
            message_dict = {}
            message_dict['content'] = []

            if message['role'] == 'user':
                message_dict['role'] = 'user'
                if 'image' in message:
                    if isinstance(message['image'], str):
                        # Single image
                        message_dict['content'].append({"type": "image"})
                        image_paths.append(message['image'])
                    elif isinstance(message['image'], list):
                        # List of images
                        for img in message['image']:
                            message_dict['content'].append({"type": "image"})
                            image_paths.append(img)
                    else:
                        raise ValueError("Invalid image type in message - should be str or List[str]")

                # Add user text message at the end
                message_dict['content'].append({"type": "text", "text": message['content']})
                input_prompt.append(message_dict)

            elif message['role'] == 'assistant':
                message_dict['role'] = 'assistant'
                message_dict['content'].append({"type": "text", "text": message['content']})
                input_prompt.append(message_dict)

            elif message['role'] == 'system':
                sys_content = message['content']
                if sys_content:
                    # Add content only if system message is found
                    print("Warning! Appending System Message.")
                    message_dict['role'] = 'system'
                    message_dict['content'].append({"type": "text", "text": sys_content})

                    input_prompt.append(message_dict)

        return {
            "prompt": input_prompt,
            "images": image_paths,
            "output_kwargs": {"device": kwargs.get('device')}
        }

    @staticmethod
    def get_tokens(prompt: list, handler: AutoProcessor, **output_kwargs) -> List[str]:
        """
        Generate tokens for the given prompt and conversation history.

        :param prompt: The current prompt to be tokenized.
        :param handler: The tokenizer used for tokenizing the prompt and history.
        :param kwargs: Additional keyword arguments, expecting 'history' which is a list of tuples (user message, assistant response).

        :return: A list of tokens generated from the combined prompt and conversation history.
        """

        chat_prompt = handler.apply_chat_template(prompt, add_generation_prompt=True)

        # Tokenize the combined text
        tokens = handler.tokenizer.tokenize(chat_prompt)

        return tokens

    @staticmethod
    def generate_outputs(prompt: list, images: List[str], model: AutoModelForVision2Seq,
                         handler: AutoProcessor, **output_kwargs) -> Tuple[Dict[str, Any], str]:
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

        # Prepare model for inference
        model = model.to(device)

        # Process images
        processed_images = []
        for image in images:
            processed_images.append(load_image(image))

        processed_prompt = handler.apply_chat_template(prompt, add_generation_prompt=True)
        inputs = handler(text=processed_prompt, images=processed_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_texts = handler.batch_decode(generated_ids, skip_special_tokens=True)

        # Process and clean response text
        response_text = generated_texts[-1]
        response_text = response_text.split("Assistant:")[-1]
        response_text = response_text.strip()

        # Format response
        response = {"response": generated_texts}

        return response, response_text
