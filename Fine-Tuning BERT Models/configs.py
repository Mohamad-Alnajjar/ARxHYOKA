from itertools import product
import os
import torch

class Configs:
    def __init__(self,
                 train_prompts = ['1'],
                 dev_prompt = '2',
                 task = 'A',
                 data_path = '/',
                 results_path = 'results/',
                 model_save_path = 'models/',
                 model_name='microsoft/mdeberta-v3-base',
                 batch_sizes=[16],
                 epochs_list=[100],
                 learning_rates=[2e-5,5e-5],
                 trainable_layers=[6],  # 'All' means all layers are trainable
                 all_prompts=['1', '2']):
        
        self.task = task
        self.data_path = data_path
        self.model_name = model_name
        self.batch_sizes = batch_sizes
        self.epochs_list = epochs_list
        self.learning_rates = learning_rates
        self.trainable_layers = trainable_layers
        self.all_prompts = all_prompts
        self.train_prompts = train_prompts
        self.dev_prompt = dev_prompt

        if self.task == 'A':
            self.trait = ['holistic']
            self.num_labels = 1  # For Task A, we have a single holistic score
        elif self.task == 'B':
            self.trait = ['relevance', 'organization', 'vocabulary', 'style', 'development', 'mechanics', 'grammar']
            self.num_labels = 7  # For Task B, we have multiple traits
        else:
            raise ValueError("Task must be either 'A' or 'B'.")
        
        self.param_grid = list(product(self.batch_sizes,self.epochs_list,self.learning_rates, self.trainable_layers))

        self.results_path = f'{results_path}/{self.task}/'
        os.makedirs(self.results_path, exist_ok=True) #if the path does not exist, create it

        self.model_save_path = f'{model_save_path}/{self.task}/'
        os.makedirs(self.model_save_path, exist_ok=True) #if the path does not exist, create it

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


