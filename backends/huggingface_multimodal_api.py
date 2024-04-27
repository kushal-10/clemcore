"""
Backend using HuggingFace transformers for ungated multimodal models.
"""
from typing import List, Dict, Tuple, Any
import torch
import backends
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForVision2Seq, IdeficsForVisionText2Text
from jinja2 import Template

logger = backends.get_logger(__name__)

def load_processor(model_spec: backends.ModelSpec) -> AutoProcessor:
    '''
    Load processor for a specific model (Example - LlavaProcessor, Kosmos2Processor) 

    :param model_spec: A dictionary that defines the model to be used, loaded from Model Registry
    :return processor: Processor for the specific model
    '''
    logger.info(f'Loading huggingface model Processor: {model_spec.model_name}')
    hf_model_str = model_spec['huggingface_id'] # Get the model name
   
    processor = AutoProcessor.from_pretrained(hf_model_str, use_fast=False, device_map="auto", verbose=False)

    return processor


def load_model(model_spec: backends.ModelSpec) -> AutoModelForVision2Seq:
    '''
    Load a specific model 

    :param model_spec: A dictionary that defines the model to be used, loaded from Model Registry
    :return model: The specific model
    '''

    logger.info(f'Start loading huggingface model weights: {model_spec.model_name}')
    hf_model_str = model_spec['huggingface_id'] # Get the model name

    if model_spec['model_name'] != 'idefics-80b-instruct': 
        model = AutoModelForVision2Seq.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto")
    else:
        model = IdeficsForVisionText2Text.from_pretrained(hf_model_str, device_map="auto", torch_dtype=torch.bfloat16)
    
    logger.info(f"Finished loading huggingface model: {model_spec.model_name}")
    
    return model

def load_image(image_path: str) -> Image:
    '''
    Load an image from a given link/directory

    :param image_path: A string that defines the link/directory of the image 
    :return image: Loaded image
    '''
    if image_path.startswith('http') or image_path.startswith('https'):
        image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    return image
    

def pad_images(images):
    '''
    Pad the images
    '''
    # Determine the maximum width and height among all images
    max_width = max(image.size[0] for image in images)
    max_height = max(image.size[1] for image in images)

    # Create and return a list of padded images
    padded_images = []
    for image in images:
        # Create a new image with a black background
        new_image = Image.new("RGB", (max_width, max_height))

        # Calculate the position to paste the image so that it's centered
        x = (max_width - image.size[0]) // 2
        y = (max_height - image.size[1]) // 2

        # Paste the original image onto the new image
        new_image.paste(image, (x, y))
        padded_images.append(new_image)

    return padded_images

def get_images(prompt_text: str, messages: list[Dict], image_placeholder: str) -> list:
    '''
    Return loaded images from messages

    :param prompt_text: A string that goes into the input of the Processor
    :param messages: A list of messages passed to the model
    :return images: A list of image locations/ PIL Images, that can be directly passed as input to the Processor.
    '''

    # Count number of image placeholders (<image>, <img>, ...) in the cleaned prompt
    num_images = prompt_text.count(image_placeholder) 
    
    # Collect image links/file locations mentioned in messages
    imgs = []
    for _, message in enumerate(messages):
        if 'image' in message:
            imgs.append(message['image'])
    if image_placeholder == "":
        return imgs

    # Check if number of image placeholders and number of images passed are valid
    if len(imgs) != num_images:
        if len(imgs) == 1:
            # If only one image is available, copy it for num_images times. 
            # For games that have single image for all turns, but the game passes only one image in the message
            imgs *= num_images
        else:
            # If the number of images doesn't match. 
            # For games that have different image at each turn, number of image in message = number of image passed.  
            raise ValueError(f"Number of images ({len(imgs)}) does not match expected count ({num_images}).\nPlease check the messages and ensure there is an image link/location for each turn")

    # Load Images
    loaded_images = [load_image(m) for m in imgs]

    return loaded_images


class HuggingfaceMultimodal(backends.Backend):
    def __init__(self):
        super().__init__()

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        return HuggingfaceMultimodalModel(model_spec)


class HuggingfaceMultimodalModel(backends.Model):

    def __init__(self, model_spec: backends.ModelSpec):
        super().__init__(model_spec)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = load_processor(model_spec)
        self.multimodal_model = load_model(model_spec)
        self.template = model_spec["custom_chat_template"]
        self.assistant_tag = model_spec["assistant"]
        self.image_placeholder = model_spec["placeholder"]
        self.padding = False
        self.IDEFICS = False
        if model_spec['model_name'] == 'idefics-80b-instruct':
            self.IDEFICS = True
        if model_spec["padding"]:
            self.padding = True

    def generate_response(self, messages: List[Dict],
                          log_messages: bool = False) -> Tuple[Any, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "user", "content": "Are there any clouds in the image? Answer with only "Yes" or "No"."},
                    {"role": "assistant", "content": "Yes"},
                    {"role": "user", "content": "This seems correct."},
                    {'role': 'user', 'content': 'Are there any chickens in the image? Answer with only "Yes" or "No".', 'image': 'games/cloudgame/resources/images/3.jpg'}
                ]
        :param model: model name
        :param log_messages: If True, raw and cleaned messages passed will be logged.
        :return: the continuation
        """

        # Get prompt by applying jinja template
        template_str = self.template
        template = Template(template_str)
        prompt_text = template.render(messages=messages)

        print("### PROMPT TEXT ###")
        print(prompt_text)

        # Get a list of images that will be passed to the Processor
        images = get_images(prompt_text, messages, self.image_placeholder)
        if self.padding:
            images = pad_images(images)

        prompt = {"inputs": prompt_text, "max_new_tokens": self.get_max_tokens(), "temeprature": self.get_temperature()}

        if not self.IDEFICS:         
            # Generate the output
            inputs = self.processor(prompt_text, images=images, return_tensors="pt").to(self.device)
            model_output = self.multimodal_model.generate(**inputs, max_new_tokens=self.get_max_tokens())
            generated_text = self.processor.batch_decode(model_output, skip_special_tokens=True)

        else:    
            idefics_input = [] #A list containing the prompt text, images specific to idefics input
            for m in messages:
                if m['role'] == 'user':
                    idefics_input.append('\nUSER: ' + m['content'])
                    if 'image' in m.keys():
                        idefics_input.append(m['image'])
                    idefics_input.append('<end_of_utterance>')
                elif m['role'] == 'assistant':
                    idefics_input.append('\nASSISTANT: ' + m['content'])        
                    idefics_input.append('<end_of_utterance>')    
            idefics_input.append('\nASSISTANT:')  

            inputs = self.processor(idefics_input, return_tensors="pt").to(self.device)
            # Generation args for Idefics
            exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
            bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
            generated_ids = self.multimodal_model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Store generated text
        response = {'response': generated_text}
        print("### GENERATED RESPONSE ###")
        print(response)

        response_text = generated_text[0].split(self.assistant_tag)[-1] # Get the last assistant response

        return prompt, response, response_text
