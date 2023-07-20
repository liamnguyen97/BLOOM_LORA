from config import Config
from prompt import Prompter
from process_analysis import DataProcess
from model_inputs import MODEL_INPUTS
from eval_and_test import EVALUATEandTEST
# from train import Trainer
from transformers import get_scheduler,TrainingArguments,Trainer,DataCollatorForLanguageModeling,DataCollatorForSeq2Seq
import torch
from contextlib import nullcontext
from torch.cuda.amp import GradScaler, autocast
import evaluate

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

    # train_dataloader, valid_dataloader = model_inputs.prepare_dataloader(train_data,
    #                                                                      valid_data,
    #                                                                      batch_size = 2)


    # Train
    metrics = evaluate.load("rouge")
    evalntest = EVALUATEandTEST(tokenizer = tokenizer,
                                metrics = metrics,
                                device = device,
                                prompter = prompter,
                                ctx = ctx)
    
    # trainer = Trainer(lr = 1e-4,
    #                   epochs = 5,
    #                   model = lora_model,
    #                   gradient_accumulation_steps = 1,
    #                   device = device,
    #                   evaluate_fn = evalntest.evaluate,
    #                   mixed_precision_dtype = mixed_precision_dtype,
    #                   scaler = scaler, 
    #                   ctx = ctx)
    
    # # checkpoint = ...
    
    # trainer.train(train_dataloader = train_dataloader,
    #               display_steps = 500,
    #               save_steps = 3000,
    #               save_name = "bloom-560m-checkpoint.pt",
    #             #   valid_dataloader = valid_dataloader,
    #               valid_dataloader= None,
    #               samples_gen = 100,
    #               samples_eval = None,
    #               gen_mode = False,
    #               checkpoint = None)
    trainer = Trainer(
        model=lora_model,
        train_dataset=train_data,
        args= TrainingArguments(
            per_device_train_batch_size=6,
            gradient_accumulation_steps=1,
            # warmup_steps=100,
            num_train_epochs=3,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=500,
            report_to="none",
            # evaluation_strategy="steps",
            # save_strategy="steps",
            # eval_steps=200,
            # save_steps=200,
            output_dir="BLOOM-alpaca",
            # save_total_limit=3,
            # load_best_model_at_end=True,
            deepspeed="ds_config_zero3.json"
            # ddp_find_unused_parameters=False if ddp else None,
        ),
        data_collator= DataCollatorForSeq2Seq(tokenizer = tokenizer,padding = True, return_tensors = "pt")
    )
    trainer.train()