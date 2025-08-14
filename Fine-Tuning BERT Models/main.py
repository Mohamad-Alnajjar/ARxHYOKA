# %% Importing necessary libraries
import argparse
import pandas as pd
import numpy as np
import warnings
import time
from train import train_model, evaluate_model, prepare_dataset
from utils import read_dataset, set_seed
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# import logging
import logging
logging.basicConfig(level=logging.INFO)
from configs import Configs

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a model for the TAQEEM shared-task.")
    parser.add_argument('--task', type=str, default='A', choices=['A', 'B'], help='Task type: A or B')
    parser.add_argument('--train_prompts', nargs='+', default=['1'], help='List of training prompts')
    parser.add_argument('--dev_prompt', type=str, default='2', help='Development prompt')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--train_and_eval', action='store_true', help='Whether to train and evaluate the model')
    parser.add_argument('--run_file_name', type=str, default='TeamID_RunID.csv', help='Name of the run file to save predictions')
    parser.add_argument('--save_run_file', action='store_true', help='Whether to save the run file with predictions')
    parser.add_argument('--train_data_path', type=str, default='/', help='The path to your training data file')

    args = parser.parse_args()
    set_seed(args.seed)  # Set the random seed for reproducibility
    print(args.train_and_eval)
    configs = Configs(task=args.task, train_prompts=args.train_prompts, dev_prompt=args.dev_prompt,data_path = args.train_data_path)

    train_set, dev_set = read_dataset(configs.task, configs.train_prompts, configs.dev_prompt,
                                        data_path=configs.data_path)
    train_set, dev_set = prepare_dataset(train_set=train_set, dev_set=dev_set, trait=configs.trait,
                                        Task=configs.task, model_name=configs.model_name,SEED= args.seed)

    logging.info(f'Train set size: {len(train_set)} examples | Dev set size: {len(dev_set)} examples')

    for batch_size,epochs,learning_rate, num_of_trainable_layers in configs.param_grid:
        logging.info(f"Fold {configs.dev_prompt} > Training {configs.model_name} on {configs.trait} with lr={learning_rate}, batch_size={batch_size}, epochs={epochs}, trainable_layers={num_of_trainable_layers}")

        exp_name = f'TAQEEM_Task_{configs.task}_Baseline_lr{learning_rate}_ba{batch_size}_ep{epochs}_tl{num_of_trainable_layers}'
        logging.info(f'Exp name: {exp_name}')
        results_path = f'{configs.results_path}/{exp_name}_TargetFold{configs.dev_prompt}.csv'
        model_save_path = f'{configs.model_save_path}/{exp_name}_TargetFold{configs.dev_prompt}'

        if args.save_run_file:
            run_file_path = f'{configs.results_path}/{args.run_file_name}'
        else:
            run_file_path = None


        start_time = time.time()

        if args.train_and_eval:
            
            train_model(train_dataset=train_set, eval_dataset=dev_set, num_labels=configs.num_labels,
                        model_save_path=model_save_path, model_name=configs.model_name, Task=configs.task,
                        batch_size=batch_size, learning_rate=learning_rate, epochs=epochs,
                        num_of_trainable_layers=num_of_trainable_layers, exp_name=exp_name
                        )

        evaluate_model(tokenized_dataset=dev_set,prompt=configs.dev_prompt,trait=configs.trait,model_save_path=model_save_path,
                        model_name=configs.model_name, batch_size=batch_size,
                        learning_rate=learning_rate,epochs=epochs, num_of_trainable_layers=num_of_trainable_layers,
                        results_save_path=results_path,save_run_file=args.save_run_file,
                        run_file_path=run_file_path)
        
        end_time = time.time()
        logging.info(f'Time taken for training and evaluation: {(end_time - start_time)/60:.2f} min')

if __name__ == "__main__":
    main()


# run the script on the command line on background
# This command runs task B baseline (Trait-specific Scoring) training on prompt 1 and using prompt 2 for evaluation, with a seed of 42, training and evaluating the model (if you want to evaluate directly then omit this argument), saving the results to a specified CSV file, and defining the path to the training data.
# python main.py --task B --train_prompts 1 --dev_prompt 2 --seed 42 --train_and_eval --run_file_name TeamID_RunID.csv --save_run_file --train_data_path /data &
