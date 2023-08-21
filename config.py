import transformers
from transformers import BloomForCausalLM, AutoTokenizer, AutoConfig ,AutoModelForCausalLM,BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
import torch
class Config:
    def __init__(self,
                 device):
        self.device = device
    
    def tokenizer(self, model_checkpoint):
        tok = AutoTokenizer.from_pretrained(model_checkpoint)
        return tok
    
    def load_pretrained_model(self, model_checkpoint):
        model = BloomForCausalLM.from_pretrained(model_checkpoint)
        
       
        return model.to(self.device)
    
    def add_lora(self, model, r: int, lora_alpha: int, lora_dropout: float):
        lora_config = LoraConfig(r = r,
                                 lora_alpha = lora_alpha,
                                 lora_dropout = lora_dropout,
                                 bias = "none",
                                 task_type = "CAUSAL_LM")
        lora_model = get_peft_model(model, lora_config)
        return lora_model
    
    def get_model_hidden_size(self,model_name):
        return AutoConfig.from_pretrained(model_name).hidden_size
    
