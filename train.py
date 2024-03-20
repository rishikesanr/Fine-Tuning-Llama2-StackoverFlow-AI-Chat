import os

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    HfArgumentParser
)
from datasets import load_dataset
import torch

import bitsandbytes as bnb
from huggingface_hub import login, HfFolder

from trl import SFTTrainer

from utils import print_trainable_parameters, find_all_linear_names

from train_args_rishikesan import ScriptArguments

from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training

#Getting the arguments from the command line
parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

#Function to fine tune and train the model
def training_function(args):
    
    #Logging to the Hugging Face Hub with token passed through the command line
    login(token=args.hf_token)

    #Setting the seed for reproducibility
    set_seed(args.seed)

    #Loading the dataset from the path passed through the command line i.e huggingface dataset paths
    data_path=args.data_path

    #Loading the dataset
    dataset = load_dataset(data_path)

    #Selecting only the first 2000 samples from the training dataset for our fine tuning
    dataset['train'] = dataset['train'].select(range(2000))

    #Defining the BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    #Getting the Pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        use_cache=False,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    #Getting the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    #Setting the padding side to right
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'

    #Preparing the model for kbit training
    model=prepare_model_for_kbit_training(model)

    #Finding all the linear names in the model
    modules=find_all_linear_names(model)
    config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=modules
    )

    #Getting the peft model
    model=get_peft_model(model, config)

    #Getting the output directory
    output_dir = args.output_dir

    #Defining the training arguments
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        bf16=False,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=args.lr_scheduler_type,
        tf32=False,
        report_to="none",
        push_to_hub=False,
        max_steps = args.max_steps
    )

    #Creating an instance of the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        dataset_text_field=args.text_field,
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_arguments
    )

    #Changing the norm layers to float32
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    print('starting training')

    #Starting the training
    trainer.train()

    print('LoRA training complete')

    #Getting the directory to save the LoRA adapters in our Hugging Face Hub
    lora_dir = args.lora_dir

    #Pushing the trained model to the Hugging Face Hub
    trainer.model.push_to_hub(lora_dir, safe_serialization=False)
    
    print("saved lora adapters")

    
#Running the training function when the script is directly run
if __name__=='__main__':
    training_function(args)

