"""
    Backend using HuggingFace transformers & ungated models.
    Uses HF tokenizers instruct/chat templates for proper input format per model.
"""
from typing import List, Dict, Tuple, Any, Union
import torch
import backends
from PIL import Image
import requests


from transformers import AutoProcessor, AutoModelForVision2Seq
import copy

from jinja2 import TemplateError, Template

logger = backends.get_logger(__name__)

def load_processor(model_spec: backends.ModelSpec) -> AutoProcessor:
    logger.info(f'Loading huggingface model Processor: {model_spec.model_name}')
    hf_model_str = model_spec['huggingface_id']

    # Load the processor
    # Further models may contain Tokenizer instead of Processor
    processor = AutoProcessor.from_pretrained(hf_model_str, device_map="auto", verbose=False)
    return processor


def load_model(model_spec: backends.ModelSpec):
    logger.info(f'Start loading huggingface model weights: {model_spec.model_name}')
    hf_model_str = model_spec['huggingface_id']

    model = AutoModelForVision2Seq.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto")
    logger.info(f"Finished loading huggingface model: {model_spec.model_name}")
    logger.info(f"Model device map: {model.hf_device_map}")
    
    return model

def load_template(model_spec: backends.ModelSpec) -> str:
    logger.info(f'loading custom template for huggingface model: {model_spec.model_name}')
    template_str = model_spec['custom_chat_template']
    return template_str

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(requests.get(image_file, stream=True).raw).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

class HuggingfaceMultimodal(backends.Backend):
    def __init__(self):
        super().__init__()

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        return HuggingfaceMultimodalModel(model_spec)


class HuggingfaceMultimodalModel(backends.Model):

    def __init__(self, model_spec: backends.ModelSpec):
        super().__init__(model_spec)
        self.processor = load_processor(model_spec)
        self.multimodal_model = load_model(model_spec)
        self.template = load_template(model_spec)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate_response(self, messages: List[Dict],
                          return_full_text: bool = False,
                          log_messages: bool = False) -> Tuple[Any, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param model: model name
        :param return_full_text: If True, whole input context is returned.
        :param log_messages: If True, raw and cleaned messages passed will be logged.
        :return: the continuation
        """
        # log current given messages list:
        if log_messages:
            logger.info(f"Raw messages passed: {messages}")

        # Get prompt by applying chat template
        template_str = self.template
        template = Template(template_str)
        
        # Temporarily handle only single instance
        messages = [messages[0]]
        
        prompt_text = template.render(messages=messages)
        prompt = {"inputs": prompt_text, "max_new_tokens": 20, "temprature": self.get_temperature()}

        # Get image
        imgs = []
        for msg_idx, message in enumerate(messages):
            if 'image' in message:
                imgs.append(message['image'])
                del messages[msg_idx]['image']

        # load image
        raw_image = load_image(imgs[0])

        inputs = self.processor(prompt_text, images=raw_image, padding=True, return_tensors="pt").to("cuda")

        model_output = self.multimodal_model.generate(**inputs, max_new_tokens=20)

        generated_text = self.processor.batch_decode(model_output, skip_special_tokens=True)

        response = {'response': generated_text}

        for text in generated_text:
            response_text = text.split("ASSISTANT:")[-1]

        return prompt, response, response_text
