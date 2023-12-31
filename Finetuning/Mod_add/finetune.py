import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GPT2Tokenizer, GPT2Model
from transformers import AutoTokenizer, AutoModelForCausalLM

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
print(device)

from utils.prompter import Prompter

prompter = Prompter()


################################### Train ###################################
def train(
    # model/data params
    base_model: str = "gpt2",
    data_path: str = "dataset.json",
    output_dir: str = "./weights",
    
    # training hyperparams
    batch_size: int = 8,
    micro_batch_size: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    weight_decay: float= 1e-2,
    cutoff_len: int = 512,
    val_set_size: int = 1696, # we don't need val in our case.
    
    # lora hyperparams
    lora_r: int = 2,
    lora_alpha: int = 2,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        'c_att',
        'c_proj'
    ],
    
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "Grokking_finetuning_LoRA_mod_additition",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    wandb_api_key: str = '',
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Finetuning_gpt2small:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    if len(wandb_api_key) > 0:
        os.environ['WANDB_API_KEY']=wandb_api_key
    #model = GPT2Model.from_pretrained(base_model)
    #tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    tokenizer.pad_token_id = 0
    
    tokenizer.padding_side = "left"  # Allow batched inference
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt
    
    #def generate_and_tokenize_prompt(example):
    #    instruction=example['instruction'] 
    #    input=example['input']
    #    output=example['output']
    #    if instruction!=None:
    #        txt='Instruction:'+str(instruction)+'\nQuestion:'+str(input)+'\n Answer:'+str(output)+'.'
    #        m=tokenizer(
    #        txt,
    #        truncation=True,
    #        max_length=cutoff_len,
    #        padding=False,
    #        return_tensors=None,)
    #        return(m)
    #    else:
    #        txt='Question:'+str(input)+'\nAnswer:'+str(output)+'.'
    #        m=tokenizer(
    #        txt,
    #        truncation=True,
    #        max_length=cutoff_len,
    #        padding=False,
    #        return_tensors=None,)
    #        return(m)


    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
       data = load_dataset("json", data_files=data_path)


############################################# Recovering from checkpoint #############################################
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    
######################################## Dataset creation and training ########################################
    #if val_set_size > 0:
    #    ###Train/val split
    #    train_val=Train[:-val_set_size]
    #    train_val=[generate_and_tokenize_prompt(x) for x in train_val]
    #    train_val=train_val.numpy()
    #    train_val=torch.tensor(train_val)
    
    #    train_data = Train[-val_set_size:]
    #    train_data=[generate_and_tokenize_prompt(x) for x in train_data]
    #    train_data=train_data.numpy()
    #    train_data=torch.tensor(train_data)

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = ( 
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    #else:
     #   train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
      #  val_data = None    

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

################################# Finetunning of the model #################################

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            weight_decay=weight_decay,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=False,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps", #if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=10,# if val_set_size > 0 else None,
            save_steps=50,
            output_dir=output_dir,
            save_total_limit=10,
            load_best_model_at_end=False if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
