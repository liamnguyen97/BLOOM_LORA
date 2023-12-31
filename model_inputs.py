import torch
import transformers

class MODEL_INPUTS:
    def __init__(self,
                 prompter,
                 tokenizer,
                 max_length: int):
        self.prompter = prompter
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def tokenize(self, prompt, add_eos_token = True):
        result = self.tokenizer(prompt,
                                truncation = True,
                                max_length = self.max_length,
                                padding = True,
                                return_tensors = None)
        if (   
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.max_length
            and add_eos_token
        ):
            
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result
        
    def generate_and_tokenize_prompt(self, dataset):
        full_prompt = self.prompter.generate_prompt(dataset["instruction"],
                                                    dataset["input"],
                                                    dataset["output"])
        
        tokenized_full_prompt = self.tokenize(full_prompt)
        return tokenized_full_prompt
    
    def prepare_dataloader(self,
                           train_data,
                           valid_data,
                           batch_size: int):
        
        train_dataloader = torch.utils.data.DataLoader(dataset = train_data,
                                                       batch_size = batch_size,
                                                       collate_fn = transformers.DataCollatorForSeq2Seq(tokenizer = self.tokenizer,
                                                                                                        padding = True,
                                                                                                        return_tensors = "pt"))
        valid_dataloader = torch.utils.data.DataLoader(dataset = valid_data,
                                                       batch_size = batch_size,
                                                       collate_fn = transformers.DataCollatorForSeq2Seq(tokenizer = self.tokenizer,
                                                                                                        padding = True,
                                                                                                        return_tensors = "pt"))
        return train_dataloader, valid_dataloader

