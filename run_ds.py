from config import Config
from prompt import Prompter
from process_analysis import DataProcess
from model_inputs import MODEL_INPUTS
from eval_and_test import EVALUATEandTEST
from trainner_ds import Trainer
import torch
from contextlib import nullcontext
from torch.cuda.amp import GradScaler, autocast
import evaluate
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
import deepspeed
import os

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    model_hidden_size = config.get_model_hidden_size("bigscience/bloom-560m")
    # distributed setup
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    train_batch_size = 2 * world_size

    
    # ds_config = {
    #         "fp16": {
    #             "enabled": True,
    #             "loss_scale": 0,
    #             "loss_scale_window": 1000,
    #             "initial_scale_power": 16,
    #             "hysteresis": 2,
    #             "min_loss_scale": 1
    #         },

    #         "optimizer": {
    #            "type": "Adam",
    #             "params": {
    #             "lr": 0.001,
    #             "betas": [
    #                 0.8,
    #                 0.999
    #             ],
    #             "eps": 1e-8,
    #             "weight_decay": 3e-7
    #             }
    #         },

    #         "scheduler": {
    #             "type": "WarmupLR",
    #             "params": {
    #                 "warmup_min_lr": "auto",
    #                 "warmup_max_lr": "auto",
    #                 "warmup_num_steps": "auto"
    #             }
    #         },
    #         "zero_optimization": {
    #             "stage": 3,
    #             "offload_param": {
    #                 "device": "none",
    #                 "pin_memory": True
    #             },
    #             "overlap_comm": True,
    #             "contiguous_gradients": True,
    #             "reduce_bucket_size": model_hidden_size * model_hidden_size,
    #             "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
    #             "stage3_param_persistence_threshold": 10 * model_hidden_size
    #         },
           
    #         "steps_per_print": 300,
    #         "train_batch_size": train_batch_size,
    #         "train_micro_batch_size_per_gpu": 1,
    #         "gradient_accumulation_steps": 1,
    #         "wall_clock_breakdown": False
    # }
    ds_config = {
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },

            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 3e-5,
                    "betas": [0.8, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },

            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 3e-5,
                    "warmup_num_steps": 500
                }
            },

            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "none",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "none",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": 1e6,
                "stage3_prefetch_bucket_size": 0.94e6,
                "stage3_param_persistence_threshold": 1e4,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True
            },

            "steps_per_print": 300,
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": 4,
            "gradient_accumulation_steps": 1,
            "wall_clock_breakdown": False
        }
    ds_engine, optimizer, train_dataloader, _ = deepspeed.initialize(model=lora_model,training_data=train_data, config_params=ds_config)
    # ds_engine.module.train()  # train
    trainer = Trainer(lr = 1e-4,
                      epochs = 3,
                      model = ds_engine,                  
                      optimizer = optimizer)
    
    # checkpoint = ...
    
    trainer.train(train_dataloader = train_dataloader,display_steps = 500)
