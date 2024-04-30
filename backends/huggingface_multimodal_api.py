"""
Backend using HuggingFace transformers for ungated multimodal models.
"""
from typing import List, Dict, Tuple, Any
import torch
import backends
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForVision2Seq, IdeficsForVisionText2Text, AutoModelForCausalLM, AutoTokenizer
from jinja2 import Template

# Define a map to load model from transformers Auto Classes
MODEL_TYPE_MAP = {
        "Idefics": IdeficsForVisionText2Text,
        "Vision2Seq": AutoModelForVision2Seq,
        "Emu2": AutoModelForCausalLM
    }

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


def load_tokenizer(model_spec: backends.ModelSpec) -> AutoTokenizer:
    '''
    Load tokenizer for a specific model.

    :param model_spec: A dictionary that defines the model to be used, loaded from Model Registry
    :return tokenizer: Tokenizer for the specific model
    '''
    logger.info(f'Loading huggingface model tokenizer: {model_spec.model_name}')
    hf_model_str = model_spec['huggingface_id']  # Get the model name

    tokenizer = AutoTokenizer.from_pretrained(hf_model_str, use_fast=False, device_map="auto", verbose=False)

    return tokenizer


def load_model(model_spec: backends.ModelSpec):
    '''
    Load a specific model 

    :param model_spec: A dictionary that defines the model to be used, loaded from Model Registry
    :return model: The specific model
    '''

    logger.info(f'Start loading huggingface model weights: {model_spec.model_name}')
    hf_model_str = model_spec['huggingface_id'] # Get the model name

    model_type = MODEL_TYPE_MAP[model_spec['model_type']] # Use the appropriate Auto class to  load the model 

    if hasattr(model_spec, 'trust_remote_code'):
        if model_spec['trust_remote_code']:
            if model_spec['model_type'] == "Emu2":
                model = model_type.from_pretrained(hf_model_str, device_map="auto", torch_dtype=torch.bfloat16,
                                                   trust_remote_code=model_spec['trust_remote_code'],
                                                   low_cpu_mem_usage=True)
            else:
                model = model_type.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto",
                                               trust_remote_code=model_spec['trust_remote_code'])
    else:
        model = model_type.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto") # Load the model

    # check if model's generation_config has pad_token_id set:
    if not model.generation_config.pad_token_id:
        # set pad_token_id to tokenizer's eos_token_id to prevent excessive warnings:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id #Same as processor.tokenizer.pad_token_id
    
    logger.info(f"Finished loading huggingface model: {model_spec.model_name}")
    logger.info(f"Device Map: {model.hf_device_map}")
    
    return model

def pad_images(images):
    '''
    Pad the images. Only used for LLaVA NeXT models
    Will be deprecated when issue https://github.com/huggingface/transformers/issues/29832 is closed
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

def get_images(messages: list[Dict]) -> list:
    '''
    Return loaded images from messages

    :param messages: A list of messages passed to the model
    :return images: A list of PIL Image objects.
    '''    
    # Collect image links/file locations mentioned in messages
    images = []
    for message in messages:
        if 'image' in message:
            images.append(message['image'])

    if not images:
        return None
    
    # Load Images
    loaded_images = []
    for img in images:
        if img.startswith('http') or img.startswith('https'):
            image = Image.open(requests.get(img, stream=True).raw).convert('RGB')
        else:
            image = Image.open(img).convert('RGB')
        loaded_images.append(image)

    return loaded_images

def generate_idefics_output(messages: list[Dict], 
                            model: IdeficsForVisionText2Text, 
                            processor: AutoProcessor,
                            max_tokens: int, 
                            device) -> list[str]:
    '''
    Return generated text from Idefics model 

    param messages: A list[Dict] type object passed to the backend containing 'role', 'content' and 'image'
    param model: Idefics model
    param processor: Idefics processor
    param device: Processing device - cuda/CPU
    '''

    #Create a list containing the prompt text and images specific to idefics input
    #Refer - https://huggingface.co/HuggingFaceM4/idefics-80b-instruct
    idefics_input = [] 
    for m in messages:
        if m['role'] == 'user':
            idefics_input.append('\nUser: ' + m['content'])
            if 'image' in m.keys():
                idefics_input.append(m['image'])
            idefics_input.append('<end_of_utterance>')
        elif m['role'] == 'assistant':
            idefics_input.append('\nAssistant: ' + m['content'])        
            idefics_input.append('<end_of_utterance>')    
    idefics_input.append('\nAssistant:')  
    idefics_input = [idefics_input]

    inputs = processor(idefics_input, add_end_of_utterance_token=False, return_tensors="pt").to(device)
 
    # Generation args for Idefics
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    max_tokens = 1024 # Default value for input max length = 20, set a high value for now 
    generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=max_tokens) 
    generated_text = processor.batch_decode(generated_ids)

    return generated_text


def generate_emu2_output(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str,
                         images, max_tokens: int):
    """
    Return generated text from Emu2 model.

    Inference is based on example code at https://huggingface.co/BAAI/Emu2-Chat and the used template is based on
    https://github.com/baaivision/Emu/blob/26710c97571cce867ad05867b955f7a63c9b8bd3/inference.py

    param messages: A list[Dict] type object passed to the backend containing 'role', 'content' and 'image'
    param model: Emu2 model
    param tokenizer: Emu2 tokenizer
    param device: Processing device - cuda/CPU
    """
    # process prompt and images into model input:
    inputs = model.build_input_ids(
        text=[prompt],
        tokenizer=tokenizer,
        image=images
    )

    # generate outputs:
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image=inputs["image"].to(torch.bfloat16),
        max_new_tokens=max_tokens,
        length_penalty=-1)

    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return output_text


class HuggingfaceMultimodal(backends.Backend):
    def __init__(self):
        super().__init__()

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        return HuggingfaceMultimodalModel(model_spec)


class HuggingfaceMultimodalModel(backends.Model):

    def __init__(self, model_spec: backends.ModelSpec):
        super().__init__(model_spec)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_spec['model_type']
        if self.model_type == "Emu2":
            self.processor = None
            self.tokenizer = load_tokenizer(model_spec)
        else:
            self.processor = load_processor(model_spec)
            self.tokenizer = None
        self.multimodal_model = load_model(model_spec)
        self.template = model_spec["custom_chat_template"]
        self.cull = model_spec["eos_to_cull"]

        self.padding = False
        self.IDEFICS = False
        if model_spec['model_name'] == 'idefics-80b-instruct':
            self.IDEFICS = True
        if model_spec["padding"]:
            self.padding = True

    def generate_response(self, messages: List[Dict]) -> Tuple[Any, Any, str]:
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
        # print(f"Input Messages: {messages}\n")
        # Get prompt by applying jinja template
        template_str = self.template
        template = Template(template_str)
        prompt_text = template.render(messages=messages)

        # Get a list of images that will be passed to the Processor
        images = get_images(messages)
        if self.padding and images:
            images = pad_images(images)

        # (f"Prompt Text: {prompt_text}\n")
        # print(f"Images Input: {images}\n")
        prompt = {"inputs": prompt_text, "max_new_tokens": self.get_max_tokens(), "temperature": self.get_temperature()}

        if self.IDEFICS:
            generated_text = generate_idefics_output(messages=messages,
                                                     model=self.multimodal_model,
                                                     processor=self.processor,
                                                     max_tokens=self.get_max_tokens(),
                                                     device=self.device)
        elif self.model_type == "Emu2":
            generated_text = generate_emu2_output(model=self.multimodal_model,
                                                  tokenizer=self.tokenizer,
                                                  prompt=prompt_text,
                                                  images=images,
                                                  max_tokens=self.get_max_tokens())
        else:
            # Generate the output
            if not images:  # If no images are present in the history + current uttereance, use tokenizer to get inputs
                inputs = self.processor.tokenizer(prompt_text, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(prompt_text, images=images, return_tensors="pt").to(self.device)
            model_output = self.multimodal_model.generate(**inputs, max_new_tokens=self.get_max_tokens())
            generated_text = self.processor.batch_decode(model_output, skip_special_tokens=True)
            

        # Store generated text
        response = {'response': generated_text}

        # print(f"Response: {response}\n")

        response_text = generated_text[0].split(self.cull)[-1] # Get the last assistant response

        return prompt, response, response_text