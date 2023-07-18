from config import Config
from prompt import Prompter
from process_analysis import DataProcess
from model_inputs import MODEL_INPUTS
from eval_and_test import EVALUATEandTEST
from train import Trainer

import torch
from contextlib import nullcontext
from torch.cuda.amp import GradScaler, autocast
import evaluate
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
import deepspeed
import os

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Mixed precision
    mixed_precision_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    ctx = torch.amp.autocast(device_type = device.type, dtype=mixed_precision_dtype)
    scaler = GradScaler()

    # Tokenizer and Model
    config = Config(device)
    tokenizer = config.tokenizer(model_checkpoint = "bigscience/bloom")
    model = config.load_pretrained_model(model_checkpoint = "bigscience/bloom-560m")
    lora_model = config.add_lora(model = model, r = 8, lora_alpha = 16, lora_dropout = 0.05)
    lora_model.to(device)

    # Prompt 
    prompter = Prompter()

    # Dataset
    data_process = DataProcess(data_path = "MBZUAI/Bactrian-X", tokenizer = tokenizer)
    dataset = data_process.load_data()

    ## print(data_process.draw(data_process.statistical(dataset, prompter)))

    splited_dataset = dataset.train_test_split(test_size = 0.1, seed = 42)

    # Model inputs
    model_inputs = MODEL_INPUTS(prompter = prompter,
                                tokenizer = tokenizer,
                                max_length = 512)
    
    train_data = splited_dataset["train"].shuffle().map(model_inputs.generate_and_tokenize_prompt)
    valid_data = splited_dataset["test"].map(model_inputs.generate_and_tokenize_prompt)

    train_data = train_data.remove_columns(["instruction", "input", "id", "output"])
    valid_data = valid_data.remove_columns(["instruction", "input", "id", "output"])

    train_data.set_format("torch")
    valid_data.set_format("torch")

    train_dataloader, valid_dataloader = model_inputs.prepare_dataloader(train_data,
                                                                         valid_data,
                                                                         batch_size = 2)


    # Train
    metrics = evaluate.load("rouge")
    evalntest = EVALUATEandTEST(tokenizer = tokenizer,
                                metrics = metrics,
                                device = device,
                                prompter = prompter,
                                ctx = ctx)
    

    model_hidden_size = config.get_model_hidden_size("bigscience/bloom-560m")
    # distributed setup
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    train_batch_size = 1 * world_size

    ds_config = {
            "fp16": {
                "enabled": True
            },
            "bf16": {
                "enabled": False
            },
            "zero_optimization": {
                "stage": 3,
                "offload_param": {
                    "device": "none",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": model_hidden_size * model_hidden_size,
                "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
                "stage3_param_persistence_threshold": 10 * model_hidden_size
            },
            "steps_per_print": 300,
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": 1,
            "wall_clock_breakdown": False
    }
    ds_engine,optimizer,train_dataloader = deepspeed.initialize(model=lora_model,training_data=train_data, config_params=ds_config)
    # ds_engine.module.train()  # train
    trainer = Trainer(lr = 1e-4,
                      epochs = 5,
                      model = ds_engine,
                      gradient_accumulation_steps = 4,
                      device = device,
                      evaluate_fn = evalntest.evaluate,
                      mixed_precision_dtype = mixed_precision_dtype,
                      scaler = scaler, 
                      ctx = ctx)
    
    # checkpoint = ...
    
    trainer.train(train_dataloader = train_dataloader,
                  display_steps = 500,
                  save_steps = 3000,
                  save_name = "bloom-560m-checkpoint.pt",
                  valid_dataloader = valid_dataloader,
                  samples_gen = 100,
                  samples_eval = None,
                  gen_mode = False,
                  deep_speed= True,
                  checkpoint = None)
