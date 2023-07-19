import torch
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_scheduler

class Trainer:
    def __init__(self,
                 lr: float,
                 epochs: int,
                 model,
                 optimizer):
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer
        
    def train(self,
              train_dataloader, 
              display_steps: int):
        
        num_update_steps_per_epoch = len(train_dataloader)       
        total_loss = 0
        num_steps = num_update_steps_per_epoch * self.epochs
        progress_bar = tqdm(range(num_steps))
        idx = 0
        current_steps = 0
        for epoch in range(self.epochs):
            
            self.model.train()
            for batch in train_dataloader:
                idx += 1
                if idx > current_steps:
                    batch = {k:v.to(self.model.device) for k, v in batch.items()}
                    
                    outputs = self.model(input_ids = batch["input_ids"],
                                            attention_mask = batch["attention_mask"],
                                            labels = batch["labels"],
                                            return_dict = True)
                    loss = outputs.loss
                    total_loss += loss.item()
                    self.model.backward(loss)  
                    self.model.step()                                        
                    
                    progress_bar.update(1)
                    current_steps += 1
                    
                    if current_steps % display_steps == 0:
                        print(f'Epoch: {epoch + 1} -- step: {current_steps} -- train_loss: {total_loss/current_steps} -- lr: {self.lr}')
                    

