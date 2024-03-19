from dataclasses import dataclass, field
import os
from typing import Optional

#Decorator for Clas ScriptArguments with dataclass
@dataclass
class ScriptArguments:

    #Hugging Face Token
    hf_token: str = field(metadata={"help": "Hugging Face Token for authentication"})

    #LLM Model name 
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf", metadata={"help": """The name of the pre-trained to use from 
                                                      the Hugging Face Model Hub"""}
    )

    #Seed for reproducing the results
    seed: Optional[int] = field(
        default=4761, metadata = {'help':'Seed for reproducibility of the results'}
    )

    #Path to the training dataset
    data_path: Optional[str] = field(
        default="./data/forums_short.json", metadata={"help": "Path to the training dataset"}
    )

    #Path to the output directory for storing logs etc
    output_dir: Optional[str] = field(
        default="output", metadata={"help": "Path to save the fine-tuned model"}
    )
    
    #Batch size for training
    per_device_train_batch_size: Optional[int] = field(
        default = 2, metadata = {"help":"Batch size for training per device"} 
    )

    #Gradient accumulation steps for training
    gradient_accumulation_steps: Optional[int] = field(
        default = 1, metadata = {"help":"Number of steps to gradient accumulation for training the model"}
    )

    #Optimizer for training such as adamw, sgd etc
    optim: Optional[str] = field(
        default = "paged_adamw_32bit", metadata = {"help":"Optimizer to use for training"}
    )

    #Frequency of the steps to save the model during training
    save_steps: Optional[int] = field(
        default = 25, metadata = {"help":"The frequency with which to save the model during training"}
    )

    #Frequency of the steps to log information during training
    logging_steps: Optional[int] = field(
        default = 1, metadata = {"help":"The frequency with which to log during training"}
    )

    #Learning rate for optimizers
    learning_rate: Optional[float] = field(
        default = 2e-4, metadata = {"help":"Learning rate for training the model"}
    )

    #Maximum gradient norm for model trainig
    max_grad_norm: Optional[float] = field (
        default = 0.3, metadata = {"help":"Maximum gradient norm for training the model"}
    )

    #Number of epochs to train the model
    num_train_epochs: Optional[int] = field (
        default = 1, metadata = {"help":"Number of epochs to train the model"}
    ) 

    #Warup ratio for model training 
    warmup_ratio: Optional[float] = field (
        default = 0.03, metadata = {"help":"Ratio of warmup steps to total steps for training the model"}
    )

    #Learning rate scheduler type for model training 
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata = {"help":"Learning rate scheduler to use for training the model"}
    ) 

    #Out Directory path to save the LoRA adapters in Hugging Face Hub
    lora_dir: Optional[str] = field(
        default = "./model/llm_hate_speech_lora", metadata = {"help":"=Directory to save the LoRA adapters in Hugging Face Hub"}
        )

    #Maximum number of step for model training steps irresepective of epochs or data
    max_steps: Optional[int] = field(
        default=-1, metadata={"help": "Maximum number of steps in model training"}
        )

    #Column that has the text data in our training dataset
    text_field: Optional[str] = field(
        default='chat_sample', metadata={"help": "The column in the dataset that has the text data"}
        )


