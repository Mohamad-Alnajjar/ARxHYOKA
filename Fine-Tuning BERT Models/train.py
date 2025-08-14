import os
import pandas as pd
import numpy as np
import torch
from utils import compute_metrics, tokenize_function, save_results, save_predictions
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, AutoConfig
)
from torch.optim import AdamW
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_dataset(train_set: pd.DataFrame, dev_set: pd.DataFrame, trait: list,
                    Task: str, model_name: str, SEED = 42):
    """    
    Prepares the dataset for training and evaluation.
    """
    train_set[trait] = train_set[trait].astype(float)
    dev_set[trait] = dev_set[trait].astype(float)

    train_dataset = Dataset.from_pandas(train_set[['essay','essay_id'] + trait])
    dev_dataset = Dataset.from_pandas(dev_set[['essay','essay_id'] + trait])

    if Task == 'A':
        if len(trait) == 1:
            trait_col = trait[0]
        else:
            raise ValueError("For Task A, trait must be a single column")

        train_dataset = train_dataset.rename_column('essay', 'text').rename_column(trait_col, 'label')
        dev_dataset = dev_dataset.rename_column('essay', 'text').rename_column(trait_col, 'label')
    elif Task == 'B':
        train_dataset = train_dataset.rename_column('essay', 'text')
        dev_dataset = dev_dataset.rename_column('essay', 'text')

        # Defensive mapping: Always create a list of the correct length (7)
        def safe_label_map(x):
            # Extract trait values in order, fill with 0.0 if missing
            label = [float(x.get(t, 0.0)) for t in trait]
            # Ensure exactly 7 elements (truncate or pad with zeros)
            label = (label + [0.0] * 7)[:7]
            return {"label": label}

        train_dataset = train_dataset.map(safe_label_map)
        dev_dataset = dev_dataset.map(safe_label_map)

        train_dataset = train_dataset.remove_columns(trait)
        dev_dataset = dev_dataset.remove_columns(trait)

    # Sanity check (can remove/comment after confirming)
    print("Sample label lengths (train):", [len(train_dataset[i]['label']) for i in range(min(10, len(train_dataset)))])
    print("Sample label lengths (dev):", [len(dev_dataset[i]['label']) for i in range(min(10, len(dev_dataset)))])

    dataset = DatasetDict({
        'train': train_dataset,
        'dev': dev_dataset
    })

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_datasets = dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=SEED)
    small_eval_dataset = tokenized_datasets["dev"].shuffle(seed=SEED)

    return small_train_dataset, small_eval_dataset

def train_model(train_dataset: Dataset, eval_dataset: Dataset,
                model_save_path: str, model_name: str, Task: str = 'A', num_labels: int = 1,
                batch_size=32, learning_rate=2e-5, epochs=100, num_of_trainable_layers=3,exp_name="TAQEEM_STL_Baseline"):

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    model.config.problem_type = "regression"

    # Set the trainable layers of the model
    if num_of_trainable_layers != 'All':
        logging.info(f'Training only the last {num_of_trainable_layers} layers of the model')
        # Freeze all parameters first
        for param in model.base_model.parameters():
            param.requires_grad = False

        # Unfreeze the last 'num_of_trainable_layers' layers of the encoder
        total_layers = len(model.base_model.encoder.layer)
        for layer_idx in range(total_layers - num_of_trainable_layers, total_layers):
            for param in model.base_model.encoder.layer[layer_idx].parameters():
                param.requires_grad = True

        # SAFE POOLER UNFREEZE: only if it exists!
        if hasattr(model.base_model, 'pooler') and model.base_model.pooler is not None:
            for param in model.base_model.pooler.parameters(): 
                param.requires_grad = True
    else: # all layers are trained (requires gradient)
        logging.info('Training all layers of the model')
        for param in model.base_model.parameters():
            param.requires_grad = True

    args = TrainingArguments(
        eval_strategy="epoch",               
        save_strategy="epoch",                
        logging_strategy="epoch",            
        save_total_limit=1,                # Keep only the latest checkpoint
        load_best_model_at_end=True,  
        metric_for_best_model="eval_qwk",   # Select the best model based on QWK
        greater_is_better=True,             # Higher QWK is better
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        lr_scheduler_type="constant",
        num_train_epochs=epochs,
        weight_decay=0.01,
        output_dir=model_save_path,   # Directory to save the last model
        logging_dir=f'./logs/{exp_name}',              # Directory for storing logs
        disable_tqdm=True                  # Disable the progress bar
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        optimizers=(AdamW(model.parameters(), lr=learning_rate), None)
    )

    logging.info(f'Training model ...')
    trainer.train()
    logging.info(f'Saving last model to {model_save_path}')
    trainer.save_model(model_save_path)

def evaluate_model(tokenized_dataset, prompt, trait, model_save_path, model_name, 
                   batch_size, learning_rate, epochs, num_of_trainable_layers, results_save_path,
                   save_run_file=True, run_file_path=None):

    model = AutoModelForSequenceClassification.from_pretrained(model_save_path).to(device)

    args = TrainingArguments(
        output_dir=model_save_path,
        per_device_eval_batch_size=batch_size
    )

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=tokenized_dataset,
        compute_metrics=compute_metrics
    )

    eval_results = trainer.evaluate()
    eval_results['prompt_id'] = prompt

    if save_run_file:
        if 'essay_id' in tokenized_dataset.column_names:
            essay_ids = tokenized_dataset['essay_id']
        predictions_output = trainer.predict(tokenized_dataset)
        predictions = predictions_output.predictions
        save_predictions(predictions, essay_ids, prompt, trait, run_file_path)
    
    save_results(learning_rate, batch_size, epochs, num_of_trainable_layers, eval_results, model_name, results_save_path, trait)
