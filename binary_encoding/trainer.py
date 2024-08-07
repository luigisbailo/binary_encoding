import importlib
import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from binary_encoding.metrics import (
    get_collapse_metrics,
    get_binarity_metrics,
    get_mahalanobis_score,
    get_same_encoding_fraction,
)
from binary_encoding.deepfool import deepfool


class Trainer():
    """
    A class that represents a trainer for a neural network classifier.

    Parameters:
    - device (torch.device): The device to use for training.
    - network (torch.nn.Module): The neural network model to train.
    - trainset (torch.utils.data.Dataset): The training dataset.
    - testset (torch.utils.data.Dataset): The testing dataset.
    - training_hypers (dict): A dictionary containing hyperparameters for training.
    - model (str): The type of model being trained \
        ('bin_enc', 'lin_pen', 'nonlin_pen', 'no_pen').
    - encoding_metrics (bool): Whether to calculate encoding metrics during training.
    - store_penultimate (bool): Whether to store penultimate layer activations after training.
    - verbose (bool): Whether to print training progress.

    Methods:
    - fit(): Trains the neural network model.
    - eval(dataset): Evaluates the neural network model on a given dataset.

    Returns:
    - res_dict_stack (dict): A dictionary containing the stacked training results.
    """

    def __init__(self, device, network, trainset, testset, training_hypers, model_name,
                 encoding_metrics=False, store_penultimate=False, verbose=True):
        self.device = device
        self.network = network
        self.trainset = trainset
        self.testset = testset
        self.training_hypers = training_hypers
        self.model_name = model_name
        self.encoding_metrics = encoding_metrics
        self.store_penultimate = store_penultimate
        self.verbose = verbose

    def fit(self):
        """
        Trains the neural network model.

        Returns:
        - res_dict_stack (dict): A dictionary containing the stacked training results.
        """
        
        torch_optim_module = importlib.import_module("torch.optim")
        
        model_ref = self.network.module if isinstance(self.network, nn.DataParallel) else self.network

        excluded_params = set(model_ref.output_layer.parameters())

        other_params = [param for name, param in model_ref.named_parameters() if param not in excluded_params]
        excluded_params = list(excluded_params)  
        params_to_update = [
            {'params': other_params, 'lr': self.training_hypers['lr']},
            {'params': excluded_params, 'lr': self.training_hypers['lr']}
        ]

        self.opt = getattr(torch_optim_module, self.training_hypers['optimizer'])(
            params_to_update,
            weight_decay=self.training_hypers['weight_decay']
        )

        self.scheduler = StepLR(
            self.opt, step_size=self.training_hypers['lr_scheduler_step_size'], gamma=self.training_hypers['lr_scheduler_gamma']
        )
        
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.training_hypers['batch_size'],
            shuffle=True
        )

        res_list = []
        res_dict_stack = {}

        gamma = self.training_hypers['gamma']
        gamma_max = 10**self.training_hypers['gamma_max_exp']
        converged = False
        convergence_thres = self.training_hypers['convergence_thres']
        
        for epoch in range(1, self.training_hypers['epochs']+1):
            
            self.network.train()

            for x_batch, y_batch in trainloader:

                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                x_output_batch, x_penultimate_batch = self.network(x_batch)

                self.opt.zero_grad()

                loss_classification = nn.CrossEntropyLoss(reduction='mean')(
                    x_output_batch, y_batch)
                loss = loss_classification

                if self.model_name == 'bin_enc':
                    loss_encoding = nn.functional.mse_loss(
                        x_penultimate_batch,
                        torch.zeros(x_penultimate_batch.shape).to(self.device),
                        reduction='mean')
                    loss = loss + loss_encoding*gamma

                loss.backward()
                
                self.opt.step()
            

            if epoch % self.training_hypers['gamma_scheduler_step'] == 0 and gamma < gamma_max:
                gamma = gamma * self.training_hypers['gamma_scheduler_factor']

            if epoch % self.training_hypers['logging'] == 0 or \
                    epoch == self.training_hypers['epochs']:
                
                if self.verbose:
                    print('Epoch', epoch)

                res_epoch = {}

                eval_train = self.eval(self.trainset)
                eval_test = self.eval(self.testset)

                res_epoch['accuracy_train'] = eval_train['accuracy']
                res_epoch['accuracy_test'] = eval_test['accuracy']

                if self.verbose:
                    print(
                        'Accuracy train: ',
                        np.around(eval_train['accuracy'], 4),
                        "\tAccuracy test:",
                        np.around(eval_test['accuracy'], 4)
                    )

                res_epoch['collapse_train'] = get_collapse_metrics(eval_train)
                res_epoch['collapse_test'] = get_collapse_metrics(eval_test)

                res_epoch['same_encoding_fraction_train'] = get_same_encoding_fraction(eval_train)
                res_epoch['same_encoding_fraction_test'] = get_same_encoding_fraction(eval_test)
                
                if self.model_name == 'bin_enc' or self.model_name == 'lin_pen':
                    res_epoch['binarity_train'] = get_binarity_metrics(eval_train)
                    res_epoch['binarity_test'] = get_binarity_metrics(eval_test)

                loader = torch.utils.data.DataLoader(
                    self.testset,
                    batch_size=1000,
                    shuffle=True
                )
                images = next(iter(loader))[0]
                perturbation_list = []
                for image in images:
                    r_tot, loop_i, label, k_i, pert_image = deepfool(
                        image,
                        self.network
                    )
                    perturbation_list.append(
                        np.linalg.norm(r_tot) /
                        np.linalg.norm(image.cpu().numpy())
                    )

                last_epoch = epoch == self.training_hypers['epochs'] 
                
                
                if eval_train['accuracy'] > convergence_thres and converged==False :
                    converged = True
                    print('converged!', epoch)
                    convergence_epoch = epoch

                if last_epoch:
                    perturbation_score_tpt = np.mean(perturbation_list)
                    mahalanobis_score_tpt = get_mahalanobis_score(eval_train, eval_test)
                
                if self.store_penultimate and last_epoch:
                    penultimate_train = eval_train['x_penultimate']
                    penultimate_test = eval_test['x_penultimate']
                    
                res_list.append(res_epoch)
            
            if epoch > self.training_hypers['lr_scheduler_start']:
                self.scheduler.step()        
                self.opt.param_groups[1]['lr'] = self.training_hypers['lr']

        for key in res_list[0].keys():

            if isinstance (res_list[0][key], dict):            
                res_dict_stack[key] = {}
 
                for key2 in res_list[0][key].keys():
                    if isinstance (res_list[0][key][key2], dict):            
                        res_dict_stack[key][key2] = {}
                        for key3 in res_list[0][key][key2].keys():
                            res_dict_stack[key][key2][key3] = np.vstack(
                                [res_epoch[key][key2][key3] for res_epoch in res_list ]
                            )
                    else:                
                        res_dict_stack[key][key2] = np.vstack(
                            [res_epoch[key][key2] for res_epoch in res_list ]
                        )                    
            else:                
                res_dict_stack[key] = np.vstack([res_epoch[key] for res_epoch in res_list])
        
        res_dict_stack['convergence_epoch'] = convergence_epoch

        res_dict_stack['mahalanobis_score_tpt'] = mahalanobis_score_tpt
        res_dict_stack['perturbation_score_tpt'] = perturbation_score_tpt

        if self.store_penultimate:
            res_dict_stack['penultimate_train'] = penultimate_train
            res_dict_stack['penultimate_test'] = penultimate_test


        return res_dict_stack

    def eval(self, dataset):
        """
        Evaluates the neural network model on a given dataset.

        Parameters:
        - dataset (torch.utils.data.Dataset): The dataset to evaluate the model on.

        Returns:
        - evaluations (dict): A dictionary containing the evaluation results.
        """

        evaluations = {}
        self.network.eval()

        with torch.no_grad():

            loader = torch.utils.data.DataLoader(dataset, batch_size=1000)

            x_output = []
            y_label = []
            x_penultimate = []

            for x_batch, y_label_batch in loader:
                x_batch = x_batch.to(self.device)
                y_label_batch = y_label_batch.to(self.device)
                x_output_batch, x_penultimate_batch = self.network(x_batch)
                y_label.append(y_label_batch)
                x_output.append(x_output_batch)
                x_penultimate.append(x_penultimate_batch)

            x_output = torch.cat(x_output)
            x_penultimate = torch.cat(x_penultimate)
            y_predicted = torch.argmax(torch.softmax(x_output, dim=-1), dim=1)
            y_label = torch.cat(y_label)

            accuracy = (y_predicted == y_label).float().mean().cpu().numpy()

            x_output = x_output.cpu().numpy()
            x_penultimate = x_penultimate.cpu().numpy()
            y_predicted = y_predicted.cpu().numpy()
            y_label = y_label.cpu().numpy()

            evaluations['x_output'] = x_output
            evaluations['x_penultimate'] = x_penultimate
            evaluations['y_predicted'] = y_predicted
            evaluations['y_label'] = y_label
            evaluations['accuracy'] = accuracy

        return evaluations
    
    
    
    
    
