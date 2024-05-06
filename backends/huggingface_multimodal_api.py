"""
Backend using HuggingFace transformers for ungated multimodal models.
"""
from typing import List, Dict, Tuple, Any
import torch
import backends
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForVision2Seq, IdeficsForVisionText2Text, AutoConfig
from jinja2 import Template

# Define a map to load model from transformers Auto Classes
MODEL_TYPE_MAP = {
        "Idefics": IdeficsForVisionText2Text,
        "Vision2Seq": AutoModelForVision2Seq
    }

FALLBACK_CONTEXT_SIZE = 256

logger = backends.get_logger(__name__)

def get_context_limit(model_spec: backends.ModelSpec) -> int:
    '''
    Get the context limit of the model

    :param model_spec: Contains definitions about the model to be used
    :return context: Context limit of the model
    '''
    hf_model_str = model_spec['huggingface_id']
    model_config = AutoConfig.from_pretrained(hf_model_str)

    if hasattr(model_config, "text_config"):
        context = model_config.text_config.max_position_embeddings
    elif hasattr(model_config, "max_sequence_length"):
        context = model_config.max_sequence_length
    else:
        context = FALLBACK_CONTEXT_SIZE
    logger.info(f"Context limit for model - {hf_model_str} is {context}")

    return context

def check_context_limit(context_size: int, prompt_tokens: list, max_new_tokens: int = 100) -> Tuple[bool, int, int, int]:
    """
    External context limit check
    :param context_size: max_sequence_length/max_position_embeddings of the model
    :param prompt_tokens: List of prompt token IDs.
    :param max_new_tokens: How many tokens to generate ('at most', but no stop sequence is defined).
    :return: Tuple with
            Bool: True if context limit is not exceeded, False if too many tokens
            Number of tokens for the given messages and maximum new tokens
            Number of tokens of 'context space left'
            Total context token limit
    """
    prompt_size = len(prompt_tokens)
    tokens_used = prompt_size + max_new_tokens  # context includes tokens to be generated
    tokens_left = context_size - tokens_used
    fits = tokens_used <= context_size
    return fits, tokens_used, tokens_left, context_size

def load_processor(model_spec: backends.ModelSpec) -> AutoProcessor:
    '''
    Load processor from AutoProcessor a specific model (Example - LlavaProcessor) 

    :param model_spec: A dictionary that defines the model to be used, loaded from Model Registry
    :return processor: Processor for the specific model
    '''
    hf_model_str = model_spec['huggingface_id'] # Get the model name
   
    if hasattr(model_spec, 'not_fast'):
        # Only used by LLaVA 1.6 34B (Throws mismatch <image> token error when not set to False)
        processor = AutoProcessor.from_pretrained(hf_model_str, use_fast=False, device_map="auto", verbose=False)
    else:
        processor = AutoProcessor.from_pretrained(hf_model_str, device_map="auto", verbose=False)
    logger.info(f'Loading Processor for model : {model_spec.model_name}')

    return processor

def load_model(model_spec: backends.ModelSpec):
    '''
    Load a specific model 

    :param model_spec: A dictionary that defines the model to be used, loaded from Model Registry
    :return model: The specific model
    '''

    logger.info(f'Start loading huggingface model weights: {model_spec.model_name}')
    hf_model_str = model_spec['huggingface_id'] # Get the model name

    model_type = MODEL_TYPE_MAP[model_spec['model_type']] # Use the appropriate Auto class to  load the model 

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

    max_tokens = 2048 # Default value for input max length = 20, set a high value for now 
    generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=max_tokens) 
    generated_text = processor.batch_decode(generated_ids)

    return generated_text


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

        self.processor = load_processor(model_spec)

        self.multimodal_model = load_model(model_spec)
        self.template = model_spec["custom_chat_template"]
        self.cull = model_spec["eos_to_cull"]

        self.padding = False
        self.IDEFICS = False
        if 'idefics' in model_spec['model_name']:
            self.IDEFICS = True
        if hasattr(model_spec, 'padding'):
            self.padding = True

        self.context_size = get_context_limit(model_spec)

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

        # Get prompt by applying jinja template
        template_str = self.template
        template = Template(template_str)
        prompt_text = template.render(messages=messages)

        # Check context limit
        prompt_tokens = self.processor.tokenizer.tokenize(prompt_text)
        context_check = check_context_limit(self.context_size, prompt_tokens, max_new_tokens=self.get_max_tokens())
        if not context_check[0]:  # if context is exceeded, context_check[0] is False
            logger.info(f"Context token limit for {self.model_spec.model_name} exceeded: "
                        f"{context_check[1]}/{context_check[3]}")
            # fail gracefully:
            raise backends.ContextExceededError(f"Context token limit for {self.model_spec.model_name} exceeded",
                                                tokens_used=context_check[1], tokens_left=context_check[2],
                                                context_size=context_check[3]) 

        # Get a list of images that will be passed to the Processor
        images = get_images(messages)
        if self.padding and images:
            images = pad_images(images)

        prompt = {"inputs": prompt_text, "max_new_tokens": self.get_max_tokens(), "temperature": self.get_temperature()}

        if self.IDEFICS:
            generated_text = generate_idefics_output(messages=messages,
                                                     model=self.multimodal_model,
                                                     processor=self.processor,
                                                     max_tokens=self.get_max_tokens(),
                                                     device=self.device)
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

        response_text = generated_text[0].split(self.cull)[-1] # Get the last assistant response

        return prompt, response, response_text