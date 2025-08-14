import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    num_labels = logits.shape[1] if len(logits.shape) > 1 else 1
    qwks = []
    qwk_sum, mse_sum, rmse_sum = 0, 0, 0

    for t in range(num_labels):
        true_labels = labels[:, t] if len(labels.shape) > 1 else labels
        pred_labels = logits[:, t] if len(logits.shape) > 1 else logits

        qwk = cohen_kappa_score(
            np.round(pred_labels).astype(int),
            np.round(true_labels).astype(int),
            weights='quadratic'
        )
        qwks.append(qwk)
        qwk_sum += qwk
        mse_sum += np.mean((pred_labels - true_labels) ** 2)
        rmse_sum += np.sqrt(np.mean((pred_labels - true_labels) ** 2))

    avg_qwk = qwk_sum / num_labels
    avg_mse = mse_sum / num_labels
    avg_rmse = rmse_sum / num_labels

    return {
        'eval_qwk': avg_qwk,
        'eval_mse': avg_mse,
        'eval_rmse': avg_rmse,
        'eval_all_qwks': qwks
    }


def save_results(learning_rate, batch_size,epochs,num_of_trainable_layers, results, model_name, save_path,trait):
    results_df = pd.DataFrame()
    
    results_df['prompt_id'] = [results['prompt_id']]
    if isinstance(trait, list):
        for t in trait:
            results_df[f'eval_qwk_{t}'] = [results['eval_all_qwks'][trait.index(t)]]
    else:
        results_df['eval_qwk'] = [results['eval_qwk']]

    results_df['eval_mse'] = [results['eval_mse']]
    results_df['eval_rmse'] = [results['eval_rmse']]
    results_df['learning_rate'] = [learning_rate]
    results_df['batch_size'] = [batch_size]
    results_df['epochs'] = [epochs]
    results_df['num_of_trainable_layers'] = [num_of_trainable_layers]
    results_df['model'] = [model_name]

    write_header = not os.path.exists(save_path)

    results_df.to_csv(save_path, mode='a', header=write_header, index=False)


def save_predictions(predictions, essay_ids, prompt, trait, results_save_path):
    """
    Save predictions to a CSV file.
    Args:
        predictions (np.ndarray): The model predictions.
        essay_ids (pd.Series): The essay IDs corresponding to the predictions.
        prompt (str): The prompt ID.
        trait (list): The traits for which predictions are made.
        results_save_path (str): The path where the results will be saved.
    """

    pred_df = pd.DataFrame(columns=['prompt_id', 'essay_id'] + trait)
    pred_df['essay_id'] = essay_ids
    pred_df['prompt_id'] = prompt
    # save the predictions to the trait columns
    for i, t in enumerate(trait):
        pred_df[t] = np.round(predictions[:, i]) if len(predictions.shape) > 1 else predictions

    pred_df.to_csv(results_save_path, index=False)


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )


def read_dataset(Task: str = 'A', train_prompts: list = ['1'], dev_prompt: list = ['2'], 
                 data_path: str = '/'):

    """ Reads the dataset for the specified task and folds."""

    if Task not in ['A', 'B']:
        raise ValueError("Task must be either 'A' or 'B'.")
    
    essays = pd.read_json(f'{data_path}/Task{Task}/TAQEEM2025_Task{Task}_essays_train.json', dtype=str)
    scores =  pd.read_csv(f'{data_path}/Task{Task}/TAQEEM2025_Task{Task}_human_scores_train.csv', dtype=str)
    scores.drop(columns=['prompt_id'], inplace=True)  # drop prompt_id column from scores

    # merge essays and scores and frop duplicate columns
    essays = essays.merge(scores, on='essay_id')

    # filter train and dev sets based on prompt_id
    train_df = essays[essays['prompt_id'].isin(train_prompts)]
    dev_df = essays[essays['prompt_id'] == dev_prompt]

    return train_df, dev_df




