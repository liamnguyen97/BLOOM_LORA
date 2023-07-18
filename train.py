import torch
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_scheduler

class Trainer:
    def __init__(self,
                 lr: float,
                 epochs: int,
                 model,
                 gradient_accumulation_steps: int,
                 device,
                 optimizer,
                 evaluate_fn,
                 mixed_precision_dtype,
                 scaler,
                 ctx):
        self.epochs = epochs
        self.model = model
        # self.optimizer = AdamW(model.parameters(), lr = lr)
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device
        self._eval = evaluate_fn
        self.mixed_precision_dtype = mixed_precision_dtype
        self.scaler = scaler
        self.ctx = ctx
        
    def train(self,
              train_dataloader, 
              display_steps: int,
              save_steps: int,
              save_name: str = None,
              save_checkpoint: bool = False,
              valid_dataloader = None,
              samples_gen: int = None,
              samples_eval: int = None,
              gen_mode: bool = False,
              deep_speed: bool = False,
              checkpoint = None):
        
        num_update_steps_per_epoch = len(train_dataloader)
        
        if checkpoint is not None and not deep_speed:
            current_steps = checkpoint["current_steps"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            num_steps = num_update_steps_per_epoch * self.epochs - current_steps
            progress_bar = tqdm(range(num_steps))
            lr_scheduler = get_scheduler("cosine",
                                         optimizer = self.optimizer,
                                         num_warmup_steps = 100,
                                         num_training_steps = num_steps)
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            self.model.load_state_dict(checkpoint["model_state_dict"])
            total_loss = checkpoint["total_loss"]
            
        else:
            current_steps = 0
            num_steps = num_update_steps_per_epoch * self.epochs
            progress_bar = tqdm(range(num_steps))
            lr_scheduler = get_scheduler("cosine",
                                         optimizer = self.optimizer,
                                         num_warmup_steps = 100,
                                         num_training_steps = num_steps)
            total_loss = 0
            
        idx = 0
        for epoch in range(self.epochs):
            
            self.model.train()
            for batch in train_dataloader:
                idx += 1
                if idx > current_steps:
                    batch = {k:v.to(self.model.device) for k, v in batch.items()}           
                    if deep_speed:
                        # self.model.module.train()
                        outputs = self.model(input_ids = batch["input_ids"],
                                                attention_mask = batch["attention_mask"],
                                                labels = batch["labels"],
                                                return_dict = True)
                        loss = outputs.loss
                        total_loss += loss.item()
                        print(f"LOSS:{loss}")
                        print(f"LOSS2:{self.optimizer.cur_scale}")
                        print(f"TOTAL LOSS:{total_loss}")
                        self.model.backward(loss)  
                        self.model.step()  
                    else:
                        self.optimizer.zero_grad()
                        with self.ctx:
                            outputs = self.model(input_ids = batch["input_ids"],
                                                attention_mask = batch["attention_mask"],
                                                labels = batch["labels"],
                                                return_dict = True)
                        loss = outputs.loss
                        total_loss += loss.item()
                        
                        loss /= self.gradient_accumulation_steps

                        if self.mixed_precision_dtype:
                            self.scaler.scale(loss).backward()
                            
                            if idx % self.gradient_accumulation_steps == 0:
                                self.scaler.step(self.optimizer)
                                lr_scheduler.step()
                                self.scaler.update()
                        else:
                            loss.backward()
                            if idx % self.gradient_accumulation_steps == 0:
                                self.optimizer.step()
                                lr_scheduler.step()
                    
                    progress_bar.update(1)
                    current_steps += 1
                    
                    if current_steps % display_steps == 0:
                        if current_steps % len(train_dataloader) == 0:
                            if valid_dataloader is not None:
                                eval_ = self._eval(self.model, valid_dataloader, samples_gen = samples_gen, samples_eval = samples_eval, gen_mode = True)
                                print(f'Epoch: {epoch + 1} -- step: {current_steps} -- train_loss: {total_loss/current_steps} -- val_loss: {eval_["loss"]}')
                                print(f'rouge1: {eval_["rouge1"]} -- rouge2: {eval_["rouge2"]} -- rougeL: {eval_["rougeL"]} -- rougeLsum: {eval_["rougeLsum"]}')
                                print("----------------------- End of epoch {} -----------------------".format(epoch + 1))
                                
                            else:
                                print(f'Epoch: {epoch + 1} -- step: {current_steps} -- train_loss: {total_loss/current_steps}') 
                                print("----------------------- End of epoch {} -----------------------".format(epoch + 1))
                        else:
                            if valid_dataloader is not None:
                                eval_ = self._eval(self.model, valid_dataloader, samples_eval = samples_eval, gen_mode = False)
                                print(f'Epoch: {epoch + 1} -- step: {current_steps} -- train_loss: {total_loss/current_steps} -- val_loss: {eval_["loss"]}')
                            else:
                                print(f'Epoch: {epoch + 1} -- step: {current_steps} -- train_loss: {total_loss/current_steps}')
                    
                    if save_checkpoint is True:
                        if current_steps % save_steps == 0:
                            print("Saving..........")
                            torch.save({"model_state_dict": self.model.state_dict(),
                                        "optimizer_state_dict": self.optimizer.state_dict(),
                                        "scaler_state_dict": self.scaler.state_dict(),
                                        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                                        "current_steps": current_steps,
                                        "total_loss": total_loss},
                                       save_name)
                            print("****** Save successfully ******")

