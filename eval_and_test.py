import torch

class EVALUATEandTEST:
    def __init__(self,
                 tokenizer,
                 device,
                 metrics,
                 prompter,
                 ctx):
        self.tokenizer = tokenizer
        self.device = device
        self.metrics = metrics
        self.prompter = prompter
        self.ctx = ctx
        
    def evaluate(self,
                 model,
                 dataset,
                 gen_mode: bool = False,
                 samples_gen: int = None,
                 samples_eval: int = None):
        model.eval()
        total_loss = 0
        predicted_texts, correct_texts = [], []
        current_gen_mode = gen_mode
        for i, batch in enumerate(dataset):
            # batch = {k:v.to(self.device) for k, v in batch.items()}
            batch = {k:v for k, v in batch.items()}
            with torch.no_grad():
                with self.ctx:
                    outputs = model(input_ids = batch["input_ids"],
                                    attention_mask = batch["attention_mask"],
                                    labels = batch["labels"],
                                    return_dict = True)
            loss = outputs.loss
            total_loss += loss.item()
            
            if current_gen_mode is True:
                outputs_gen = model.generate(input_ids = batch["input_ids"],
                                             attention_mask = batch["attention_mask"],
                                             top_k = 40,
                                             no_repeat_ngram_size = 3,
                                             num_beams = 1,
                                             max_new_tokens = 256,
                                             bos_token_id = self.tokenizer.bos_token_id,
                                             eos_token_id = self.tokenizer.eos_token_id,
                                             pad_token_id = self.tokenizer.pad_token_id,
                                             early_stopping = True)
                
                generate_batch = self.tokenizer.batch_decode(outputs_gen, skip_special_tokens = True)
                correct_batch = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens = True)
                
                for j in len(generate_batch):       
                    prompt = generate_batch[j]
                    response = self.prompter.get_response(prompt)
                    generate_batch[j] = response
                        
                for k in len(correct_batch):
                    prompt = correct_batch[k]
                    response = self.prompter.get_response(prompt)
                    correct_batch[k] = response
                    
                predicted_texts += generate_batch
                correct_texts += correct_batch
                
            if samples_gen is not None:
                if i >= samples_gen:
                    current_gen_mode = False
            
            if samples_eval is not None:
                if i >= samples_eval:
                    break
                
        if gen_mode is True:
            rouge = self.metrics.compute(predictions = predicted_texts,
                                         references = correct_texts)
        
            return {"rouge1": rouge["rouge1"],
                    "rouge2": rouge["rouge2"],
                    "rougeL": rouge["rougeL"],
                    "rougeLsum": rouge["rougeLsum"],
                    "loss": total_loss/(samples_eval + 1 if samples_eval is not None else len(dataset))}
     
        else:
            return {"loss": total_loss/(samples_eval + 1 if samples_eval is not None else len(dataset))}
    
    def test(self,
             model,
             instruction: str,
             input: str = None,
             label: str = None):
        
        prompt = self.prompter.generate_prompt(instruction = instruction, input = input)
        inputs = self.tokenizer(prompt, return_tensors = "pt")
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs = {k: v for k, v in inputs.items()}
        outputs = model.generate(input_ids = inputs["input_ids"],
                                 attention_mask = inputs["attention_mask"],
                                 max_new_tokens = 256,
                                 no_repeat_ngram_size = 3,
                                 num_beams = 1,
                                 top_k = 40,
                                 bos_token_id = self.tokenizer.bos_token_id,
                                 eos_token_id = self.tokenizer.eos_token_id,
                                 pad_token_id = self.tokenizer.pad_token_id,
                                 early_stopping = True)
        text = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
        response = self.prompter.get_response(text)
        if label is not None:
            return {"label": label,
                    "response": response}
        else:
            return {"response": response}
        
