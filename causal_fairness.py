import os
import sys
import os.path as osp
import argparse
import cv2
import numpy as np
from glob import glob
import random
import time
import csv
import h5py
import copy
import pandas as pd
import importlib
from functools import partial
from omegaconf import OmegaConf
from tqdm import tqdm
from rich.progress import track
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import torchvision
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


from utils.gaze_utils import draw_gaze, angular_error, AverageMeter
from utils.args import str2bool
from utils.image_transform import my_normalize, my_denormalize, recover_image, create_grid

from datetime import datetime

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer(nn.Module):

    def __init__(self, model, train_loader, test_loader, output_dir=None, augment=False, lambda_values=None, test_type="baseline", confound_attr="average_eye_height"):
        super().__init__()

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        # self.causal_model = model
        self.augment = augment
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        # self.causal_model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)

        self.start_epoch = 0
        self.epochs = 15
        self.train_iter = 0
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ckpt_dir = osp.join(self.output_dir, 'ckpt')
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.tensorboard_dir = osp.join(self.output_dir, 'tensorboard')
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        
        # Set lambda values with default fallback
        self.lambda_values = {
            'cf': 0.0,      # Counterfactual loss weight
            'cf_bg': 0.0,   # Background removal loss weight
            'dc': 0.0        # Weighted loss weight
        }
        
        # Override default lambda values if provided
        if lambda_values:
            for key, value in lambda_values.items():
                if key in self.lambda_values:
                    self.lambda_values[key] = value
        
        # Print lambda values
        print("Lambda Values:")
        for key, value in self.lambda_values.items():
            print(f"{key}: {value}")

        # Set the test type
        self.test_type = test_type
        self.confound_attr = confound_attr

        # Fairness evaluation variables
        self.pred_gaze_list = []
        self.gaze_var_list = []
        self.angle_threshold = 5  # Consider predictions within 5 degrees as correct
        
        # CSV file for fairness metrics
        self.fairness_csv_path = 'output_ours_360.csv'#osp.join(self.output_dir, 'output_ours_360.csv')
        # Check if file exists, if not create with headers
        if not osp.exists(self.fairness_csv_path):
            with open(self.fairness_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Test Type', 'Dataset Combination', 'Threshold', 'Group Pair', 
                                 'ΔTPR (Equal Opportunity)', 'Disparate Impact'])

    def load_checkpoint(self, checkpoint_path):
        """
        Load model and optimizer state from a checkpoint.
        
        Args:
            checkpoint_path (str): The path to the checkpoint file.
        """
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optim_state'])
        self.scheduler.load_state_dict(checkpoint['schedule_state'])
        self.start_epoch = checkpoint['epoch']
        print("Checkpoint loaded successfully.")

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.train_one_epoch(epoch)
            error = self.test(epoch)
            

            if epoch + 1 == 15:  # Saves only at the last epoch when epochs is 15
                add_file_name = 'epoch_' + str(epoch+1).zfill(2) + '_error=' + str(round(error, 2))
                self.save_checkpoint(
                    {'epoch': epoch + 1,
                    'model_state': self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
                    'optim_state': self.optimizer.state_dict(),
                    'schedule_state': self.scheduler.state_dict()
                    }, add=add_file_name
                )
        

    def one_iteration(self, data, tag='train'):
        l1_criterion = nn.L1Loss()
        input_var = data['image'].float().to(self.device)
        gaze_var = data['gaze'].float().to(self.device)
        input_var_zero = torch.zeros_like(input_var)
        input_var_rf = data.get('bg_image', torch.zeros_like(input_var)).float().to(self.device)
        # Forward pass for main model
        output_dict, output_dict_zero, output_dict_rf = self.model(input_var, input_var_zero, input_var_rf)
        pred_gaze = output_dict['pred_gaze']
        pred_gaze_zero = output_dict_zero['pred_gaze']
        pred_gaze_rf = output_dict_rf['pred_gaze']

        # Save predictions for fairness evaluation
        self.pred_gaze_list.append(pred_gaze.cpu().detach().numpy())
        self.gaze_var_list.append(gaze_var.cpu().detach().numpy())

        # Standard L1 Loss
        loss_gaze = l1_criterion(pred_gaze, gaze_var)
        zero_input_loss = l1_criterion(pred_gaze_zero, gaze_var)
        rf_input_loss = l1_criterion(pred_gaze_rf, gaze_var)
        
        # Weighted Loss based on the selected confound attribute
        if self.confound_attr in data:
            confound = data[self.confound_attr].float().to(self.device)
            weights = 1 / (confound + 1e-6)  # Avoid division by zero
            weights = weights / weights.sum()
            weighted_loss_scalar = (loss_gaze * weights).mean()
        else:
            weighted_loss_scalar = torch.tensor(0.0).to(self.device)
        
        # Combine Losses
        total_loss = (
            loss_gaze + 
            self.lambda_values['cf'] * zero_input_loss + 
            self.lambda_values['cf_bg'] * rf_input_loss + 
            self.lambda_values['dc'] * weighted_loss_scalar
        )
        
        # Calculate error
        error_gaze = np.mean(angular_error(pred_gaze.cpu().data.numpy(), gaze_var.cpu().data.numpy()))

        # Tensorboard logging
        if self.train_iter!=0 and self.train_iter % 10 == 0:
            self.writer.add_scalar(f'{tag}/loss_gaze', loss_gaze.item(), self.train_iter)
            self.writer.add_scalar(f'{tag}/total_loss', total_loss.item(), self.train_iter)
            self.writer.add_scalar(f'{tag}/error_gaze', error_gaze.item(), self.train_iter)
            log_img = torchvision.utils.make_grid(input_var[:8], nrow=4, normalize=True)   
            self.writer.add_image(f'{tag}/images', log_img, self.train_iter)
        
        self.train_iter += 1
        return total_loss, error_gaze, weighted_loss_scalar
    
    def train_one_epoch(self, epoch):
        print(f'Epoch: {epoch + 1} / {self.epochs}')
        self.model.train()
        for i, data in enumerate(track(self.train_loader, description='Training', transient=True)):
            total_loss, _, _ = self.one_iteration(data)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
        # Clear predictions lists after each epoch
        self.pred_gaze_list.clear()
        self.gaze_var_list.clear()
            
        self.scheduler.step()
    
    def calculate_metrics(self, pred_angles, true_angles):
        """
        Calculate TPR, FPR, PPV, probability of positive prediction, and AUC
        """
        errors = angular_error(pred_angles, true_angles)
        
        # Consider predictions within threshold as positive
        positives = (errors <= self.angle_threshold)
        
        # Calculate the correct metrics
        # Total samples
        total_predictions = len(errors)
        
        # True positive rate should be TP / total_predictions
        # This represents the proportion of samples correctly predicted within threshold
        tpr = np.sum(positives) / total_predictions if total_predictions > 0 else 0
        
        # False positive rate isn't accurate here - in gaze estimation all predictions 
        # are "positive attempts", so this is actually 1-tpr
        fpr = 1 - tpr
        
        # Probability of positive prediction (accuracy)
        pos_prob = np.sum(positives) / total_predictions if total_predictions > 0 else 0
        
        # Calculate ROC curve points and AUC
        thresholds = np.linspace(0, 180, 1000)  # Range of possible angle thresholds
        fprs, tprs = [], []
        for thresh in thresholds:
            curr_positives = (errors <= thresh)
            tp_rate = np.sum(curr_positives) / total_predictions if total_predictions > 0 else 0
            tprs.append(tp_rate)
            fprs.append(1 - tp_rate)  # 1-TPR
        
        # Calculate AUC using trapezoidal rule
        auc = np.trapz(tprs, fprs)
        
        return {
            'tpr': tpr,
            'fpr': fpr,
            'pos_prob': pos_prob,
            'auc': auc,
            'roc_curve': (fprs, tprs)
        }

    def calculate_group_fairness(self, pred_angles_0, true_angles_0, pred_angles_1, true_angles_1):
        """
        Calculate ΔTPR and Disparate Impact between groups
        """
        # Calculate metrics for group 0
        metrics_0 = self.calculate_metrics(pred_angles_0, true_angles_0)
        # Calculate metrics for group 1
        metrics_1 = self.calculate_metrics(pred_angles_1, true_angles_1)
        
        # Calculate absolute difference in TPR (Equal Opportunity)
        delta_tpr = abs(metrics_0['tpr'] - metrics_1['tpr'])
        
        # Calculate Disparate Impact
        # Standard DI = min(P(Y^|S=1)/P(Y^|S=0), P(Y^|S=0)/P(Y^|S=1))
        # Handle division by zero with epsilon
        epsilon = 1e-10
        prob_ratio_1 = metrics_1['pos_prob'] / (metrics_0['pos_prob'] + epsilon)
        prob_ratio_2 = metrics_0['pos_prob'] / (metrics_1['pos_prob'] + epsilon)
        disparate_impact = min(prob_ratio_1, prob_ratio_2)
        
        # Ensure disparate impact is between 0 and 1
        disparate_impact = min(max(disparate_impact, 0), 1)
        
        return delta_tpr, disparate_impact, metrics_0['auc'], metrics_1['auc']

    def write_fairness_metrics_to_csv(self, dataset_combination, threshold, group_pair, delta_tpr, disparate_impact):
        """
        Write fairness metrics to CSV file
        
        Args:
            dataset_combination (str): The dataset combination being used (e.g., 'xgaze_to_mpii')
            threshold (float): The threshold used for evaluation
            group_pair (str): The group pair being compared (e.g., '0_1')
            delta_tpr (float): Delta TPR value (Equal Opportunity)
            disparate_impact (float): Disparate Impact value
        """
        with open(self.fairness_csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                self.test_type,
                dataset_combination, 
                threshold, 
                group_pair, 
                f"{delta_tpr:.4f}", 
                f"{disparate_impact:.4f}"
            ])

    def test(self, epoch):
        errors_gaze = AverageMeter()
        total_losses = AverageMeter()
        all_predictions = []
        all_ground_truth = []
        all_races = []

        self.model.eval()
        for i, data in enumerate(track(self.test_loader, description='Testing', transient=True)):
            with torch.no_grad():
                total_loss, error_gaze, _ = self.one_iteration(data, 'test')
                errors_gaze.update(error_gaze.item(), data['image'].size(0))
                total_losses.update(total_loss.item(), data['image'].size(0))
                
                # Store race data if available
                if 'race' in data:
                    all_races.append(data['race'].cpu().numpy())

        print(f'Epoch: {epoch + 1}, Error gaze: {errors_gaze.avg}, Total Loss: {total_losses.avg}')
        self.model.train()

        self.writer.add_scalar('test/epoch_error_gaze', errors_gaze.avg, epoch + 1)
        self.writer.add_scalar('test/epoch_total_loss', total_losses.avg, epoch + 1)

        with open(osp.join(self.output_dir, 'test_results.txt'), 'a') as f:
            f.write('test on epoch {}, error: {}, total_loss: {}\n'.format(epoch + 1, errors_gaze.avg, total_losses.avg))
        
        # Perform fairness evaluation if race data is available
        if len(all_races) > 0:
            # Concatenate all arrays
            all_predictions = np.concatenate(self.pred_gaze_list, axis=0)
            all_ground_truth = np.concatenate(self.gaze_var_list, axis=0)
            all_races = np.concatenate(all_races, axis=0)
            
            # Clear lists for next epoch
            self.pred_gaze_list.clear()
            self.gaze_var_list.clear()

            # Define different thresholds to test
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5]
            
            # Get unique races
            unique_races = np.unique(all_races)
            
            # Get dataset combination
            dataset_combination = self.output_dir.split('/')[-1].replace('_ours', '')
            
            # For each threshold
            for threshold in thresholds:
                print(f'\nResults for threshold {threshold}°:')
                self.angle_threshold = threshold
                
                # Calculate metrics for each pair of racial groups
                for i in range(len(unique_races)):
                    for j in range(i+1, len(unique_races)):
                        race1_mask = (all_races == unique_races[i])
                        race2_mask = (all_races == unique_races[j])
                        
                        # Get predictions for each racial group
                        pred_angles_race1 = all_predictions[race1_mask]
                        true_angles_race1 = all_ground_truth[race1_mask]
                        pred_angles_race2 = all_predictions[race2_mask]
                        true_angles_race2 = all_ground_truth[race2_mask]
                        
                        # Calculate fairness metrics
                        delta_tpr, disparate_impact, auc_race1, auc_race2 = self.calculate_group_fairness(
                            pred_angles_race1, true_angles_race1,
                            pred_angles_race2, true_angles_race2
                        )
                        
                        # Create group pair string
                        group_pair = f"{unique_races[i]}_{unique_races[j]}"
                        
                        # Write metrics to CSV
                        self.write_fairness_metrics_to_csv(
                            dataset_combination,
                            threshold,
                            group_pair,
                            delta_tpr,
                            disparate_impact
                        )
                        
                        print(f'Racial groups {unique_races[i]} vs {unique_races[j]}:')
                        print(f'ΔTPR (Equal Opportunity): {delta_tpr:.4f}')
                        print(f'Disparate Impact: {disparate_impact:.4f}')
                        print(f'AUC (Race {unique_races[i]}): {auc_race1:.4f}')
                        print(f'AUC (Race {unique_races[j]}): {auc_race2:.4f}')
                        
                        # Save metrics to tensorboard with threshold information
                        self.writer.add_scalar(f'fairness/delta_tpr_race_{unique_races[i]}_{unique_races[j]}_thresh_{threshold}', 
                                            delta_tpr, epoch + 1)
                        self.writer.add_scalar(f'fairness/disparate_impact_race_{unique_races[i]}_{unique_races[j]}_thresh_{threshold}', 
                                            disparate_impact, epoch + 1)
        
        # Check if this is a gender-specific experiment
        if "male" in self.output_dir or "female" in self.output_dir:
            # Save predictions and ground truths for later comparison
            if len(self.pred_gaze_list) > 0 and len(self.gaze_var_list) > 0:
                all_predictions = np.concatenate(self.pred_gaze_list, axis=0)
                all_ground_truth = np.concatenate(self.gaze_var_list, axis=0)
                
                # Save for cross-experiment comparison
                np.save(osp.join(self.output_dir, f'pred_gaze_epoch_{epoch+1}.npy'), all_predictions)
                np.save(osp.join(self.output_dir, f'gaze_var_epoch_{epoch+1}.npy'), all_ground_truth)
                
                # If this is the final epoch (15), perform gender fairness evaluation
                if epoch + 1 == 15:
                    # Try to find the opposite gender experiment results
                    current_dir = self.output_dir
                    if "male" in current_dir:
                        opposite_dir = current_dir.replace("male", "female")
                    else:
                        opposite_dir = current_dir.replace("female", "male")
                    
                    if os.path.exists(opposite_dir):
                        self.evaluate_gender_fairness(opposite_dir)
                    else:
                        self.evaluate_gender_fairness()
        
        return errors_gaze.avg

    def save_checkpoint(self, state, add=None):
        """
        Save a copy of the model
        """
        if add is not None:
            filename = add + '.pth.tar'
        else:
            filename = 'ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        print('save file to: ', ckpt_path)

    def evaluate_gender_fairness(self, other_results_dir=None):
        """
        Evaluate fairness across gender-specific test datasets.
        
        Args:
            other_results_dir (str, optional): Path to the results directory of the opposite gender experiment
        """
        # Identify current gender from experiment name
        current_gender = None
        if "male" in self.output_dir:
            current_gender = "male"
        elif "female" in self.output_dir:
            current_gender = "female"
        
        if not current_gender:
            print("No gender-specific experiment detected. Skipping gender fairness evaluation.")
            return
        
        print(f"\nPerforming gender fairness evaluation for {current_gender} dataset...")
        
        # If we're evaluating a single gender dataset without comparison
        if not other_results_dir:
            # Load predictions and ground truth for current gender
            all_predictions = np.concatenate(self.pred_gaze_list, axis=0)
            all_ground_truth = np.concatenate(self.gaze_var_list, axis=0)
            
            # Calculate metrics for current gender
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5]
            for threshold in thresholds:
                print(f'\nResults for threshold {threshold}°:')
                self.angle_threshold = threshold
                
                metrics = self.calculate_metrics(all_predictions, all_ground_truth)
                
                print(f'Gender: {current_gender}')
                print(f'TPR: {metrics["tpr"]:.4f}')
                print(f'FPR: {metrics["fpr"]:.4f}')
                print(f'Pos Probability: {metrics["pos_prob"]:.4f}')
                print(f'AUC: {metrics["auc"]:.4f}')
                
                # Log metrics with gender information
                dataset_combination = self.output_dir.split('/')[-1].replace('_ours', '')
                with open(osp.join(self.output_dir, 'gender_metrics.txt'), 'a') as f:
                    f.write(f'\nThreshold {threshold}°, Gender: {current_gender}:\n')
                    f.write(f'TPR: {metrics["tpr"]:.4f}, FPR: {metrics["fpr"]:.4f}\n')
                    f.write(f'Pos Probability: {metrics["pos_prob"]:.4f}, AUC: {metrics["auc"]:.4f}\n')
            
            return
        
        # For cross-experiment comparison
        # Load opposite gender results
        opposite_gender = "female" if current_gender == "male" else "male"
        try:
            # Try to load predictions from the other gender experiment
            opposite_predictions = np.load(osp.join(other_results_dir, 'pred_gaze_epoch_15.npy'))
            opposite_ground_truth = np.load(osp.join(other_results_dir, 'gaze_var_epoch_15.npy'))
            
            print(f"Loaded {opposite_gender} data for comparison")
            
            # Calculate cross-gender fairness metrics
            current_predictions = np.concatenate(self.pred_gaze_list, axis=0)
            current_ground_truth = np.concatenate(self.gaze_var_list, axis=0)
            
            # Define thresholds for evaluation
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5]
            
            # Create CSV for gender fairness metrics if it doesn't exist
            gender_fairness_csv_path = osp.join(self.output_dir, 'gender_fairness_metrics.csv')
            if not osp.exists(gender_fairness_csv_path):
                with open(gender_fairness_csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Test Type', 'Dataset Combination', 'Threshold', 
                                    'Male TPR', 'Female TPR', 'ΔTPR', 'Male AUC', 'Female AUC', 'Disparate Impact'])
            
            # For each threshold
            for threshold in thresholds:
                print(f'\nResults for threshold {threshold}°:')
                self.angle_threshold = threshold
                
                # Calculate metrics for both genders
                metrics_male = self.calculate_metrics(
                    current_predictions if current_gender == "male" else opposite_predictions,
                    current_ground_truth if current_gender == "male" else opposite_ground_truth
                )
                
                metrics_female = self.calculate_metrics(
                    current_predictions if current_gender == "female" else opposite_predictions,
                    current_ground_truth if current_gender == "female" else opposite_ground_truth
                )
                
                # Calculate gender fairness metrics
                delta_tpr = abs(metrics_male['tpr'] - metrics_female['tpr'])
                
                # Calculate Disparate Impact
                epsilon = 1e-10
                prob_ratio_1 = metrics_female['pos_prob'] / (metrics_male['pos_prob'] + epsilon)
                prob_ratio_2 = metrics_male['pos_prob'] / (metrics_female['pos_prob'] + epsilon)
                disparate_impact = min(prob_ratio_1, prob_ratio_2)
                disparate_impact = min(max(disparate_impact, 0), 1)
                
                # Log metrics 
                dataset_combination = self.output_dir.split('/')[-1].replace('_ours', '')
                
                # Write to CSV
                with open(gender_fairness_csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        self.test_type,
                        dataset_combination,
                        threshold,
                        f"{metrics_male['tpr']:.4f}",
                        f"{metrics_female['tpr']:.4f}",
                        f"{delta_tpr:.4f}",
                        f"{metrics_male['auc']:.4f}",
                        f"{metrics_female['auc']:.4f}",
                        f"{disparate_impact:.4f}"
                    ])
                
                # Print results
                print(f'Male TPR: {metrics_male["tpr"]:.4f}, Female TPR: {metrics_female["tpr"]:.4f}')
                print(f'ΔTPR (Equal Opportunity): {delta_tpr:.4f}')
                print(f'Male AUC: {metrics_male["auc"]:.4f}, Female AUC: {metrics_female["auc"]:.4f}')
                print(f'Disparate Impact: {disparate_impact:.4f}')
                
        except (FileNotFoundError, ValueError) as e:
            print(f"Could not load opposite gender data: {e}")
            print("Performing single gender evaluation instead")
            # Fall back to single gender evaluation
            self.evaluate_gender_fairness()

def build_model_from_cfg(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    """
    cfg is like:f
        type: networks.GazeResNet.GazeRes18
        params: {}
    """
    module, cls = cfg['type'].rsplit(".", 1)
    MODEL = getattr(importlib.import_module(module, package=None), cls)
    model = MODEL(**cfg.get("params", dict()))
    return model
   

from utils.util import instantiate_from_config
if __name__ == '__main__':
    this_dir = os.path.dirname(os.path.realpath(__file__))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str,  help='name of the experiment')
    parser.add_argument('--model_cfg_path', help='the path to the config file of the model', default=f'{this_dir}/configs/models/causalres18.yaml')
    parser.add_argument('--output_dir', help='the path to the output directory', default=f'{this_dir}/logs_ours_average')
    
    # Add arguments for lambda values
    parser.add_argument('--lambda_cf', type=float, default=0.0, help='Counterfactual loss weight')
    parser.add_argument('--lambda_cf_bg', type=float, default=0.0, help='Background removal loss weight')
    parser.add_argument('--lambda_dc', type=float, default=0.0, help='Weighted loss weight')
    
    # Add arguments for loading pretrained weights and evaluation-only mode
    parser.add_argument('--checkpoint', type=str, default=None, help='path to the checkpoint file for testing')
    parser.add_argument('--eval_only', action='store_true', help='only run evaluation without training')
    
    # Add argument for test type
    parser.add_argument('--test_type', type=str, default='baseline', help='Type of test being performed (e.g., baseline, causal, etc.)')
    parser.add_argument('--confound_attr', type=str, default='average_eye_height', help='Dataset key used for the DC weighted loss')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for train/test loaders')
    parser.add_argument('--num_workers', type=int, default=18, help='Number of DataLoader workers')
    
    args = parser.parse_args()

    set_seed(42)
    
    output_dir = osp.join(args.output_dir, args.exp_name + '_ours')

    transform_mean_std = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    data_location = OmegaConf.load('./configs/data_path.yaml')
    data_cfg = {
        'xgaze': './configs/datasets/xgaze.yaml',
        'xgaze_full_light': './configs/datasets/xgaze_full_light.yaml',
        'xgaze_cam0': './configs/datasets/xgaze_cam0.yaml',
        'xgaze_full_light_cam0': './configs/datasets/xgaze_full_light_cam0.yaml',
        'mpii': './configs/datasets/mpii.yaml',
        'gazecapture_test': './configs/datasets/gazecapture.yaml',
        'xgaze_gender_balance': './configs/datasets/xgaze_gender_balance.yaml',
        'xgaze_eye_height_balance': './configs/datasets/xgaze_eye_height_balance.yaml',
        'mpii_nv': './configs/datasets/mpii_nv.yaml',
        'xgaze_60':'./configs/datasets/xgaze_60.yaml',
        'xgaze_20':'./configs/datasets/xgaze_20.yaml',
        'gaze360_train': './configs/datasets/gaze360.yaml',
        'gaze360_test': './configs/datasets/gaze360_test.yaml',
        'eyediap_cs': './configs/datasets/eyediap_cs.yaml',
        'eyediap_ft': './configs/datasets/eyediap_ft.yaml',
        'mpii_male': './configs/datasets/mpii_male.yaml',
        'mpii_female': './configs/datasets/mpii_female.yaml',
    }

    exps = {
        'gaze360_to_mpii':['gaze360_train', 'mpii'],
        'gaze360_to_xgaze':['gaze360_train', 'xgaze'],
        'gaze360_to_gaze360':['gaze360_train', 'gaze360_test'],
        'xgaze_to_mpii': ['xgaze','mpii'],
        'xgaze_to_gaze360': ['xgaze', 'gaze360_test'],
        'xgaze_to_gazecapture': ['xgaze','gazecapture_test'], 
        'mpii_to_mpii': ['mpii','mpii'],
        'mpii_to_xgaze': ['mpii','xgaze'],
        'mpii_to_gazecapture': ['mpii','gazecapture_test'],
        'mpii_to_gaze360': ['mpii','gaze360_test'],
        'mpii_to_mpii_female': ['mpii','mpii_female'],
        'mpii_to_mpii_male': ['mpii','mpii_male'],
        'mpii_to_xgaze_female': ['mpii','xgaze_gender_balance'],
        'mpii_to_xgaze_male': ['mpii','xgaze_male'],
        'mpii_to_eyediap_cs': ['mpii','eyediap_cs'],
        'mpii_to_eyediap_ft': ['mpii','eyediap_ft'],
        'xgaze_to_xgaze': ['xgaze','xgaze'],
        'xgaze_to_eyediap_cs': ['xgaze','eyediap_cs'],
        'xgaze_to_eyediap_ft': ['xgaze','eyediap_ft'],
        'xgaze_to_mpii_male': ['xgaze', 'mpii_male'],
        'xgaze_to_mpii_female': ['xgaze', 'mpii_female'],
        'gaze360_to_eyediap_cs': ['gaze360_train', 'eyediap_cs'],
        'gaze360_to_eyediap_ft': ['gaze360_train', 'eyediap_ft'],
        'gaze360_to_gazecapture': ['gaze360_train', 'gazecapture_test'],
        'xgaze_to_xgaze_female': ['xgaze','xgaze_gender_balance'],
        'xgaze_to_xgaze_male':['xgaze','xgaze_male'],
        'gaze360_to_mpii_male':['gaze360_train','mpii_male'],
        'gaze360_to_mpii_female':['gaze360_train','mpii_female'],
        'gaze360_to_xgaze_female':['gaze360_train','xgaze_gender_balance'],
        'gaze360_to_xgaze_male':['gaze360_train','xgaze_male'],
        }
    
    train_data_name = exps[args.exp_name][0]
    train_cfg = OmegaConf.load(data_cfg[train_data_name])
    train_cfg['params']['transform_Normalize'] = transform_mean_std
    train_cfg['params']['dataset_path'] = data_location['xgaze' if 'xgaze' in train_data_name else train_data_name]
    train_cfg['params']['confound_attr'] = args.confound_attr
    train_dataset = instantiate_from_config(train_cfg)

    test_data_name = exps[args.exp_name][1]
    test_cfg = OmegaConf.load(data_cfg[test_data_name])
    test_cfg['params']['transform_Normalize'] = transform_mean_std
    test_cfg['params']['dataset_path'] = data_location['xgaze' if 'xgaze' in test_data_name else test_data_name]
    test_cfg['params']['confound_attr'] = args.confound_attr
    test_datset = instantiate_from_config(test_cfg)
    
    print(f'train_dataset {train_data_name} num of samples: ', len(train_dataset))
    print(f'test_dataset {test_data_name} num of samples: ', len(test_datset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_datset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = build_model_from_cfg(args.model_cfg_path)
    # print(model)
    # summary(model)

    os.makedirs(output_dir, exist_ok=True)
    OmegaConf.save(transform_mean_std, osp.join(output_dir, 'transform_mean_std.yaml'))

    # Prepare lambda values dictionary
    lambda_values = {
        'cf': args.lambda_cf,
        'cf_bg': args.lambda_cf_bg,
        'dc': args.lambda_dc
    }

    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        output_dir=output_dir, 
        lambda_values=lambda_values,  # Pass lambda values to Trainer
        test_type=args.test_type,  # Pass test type to Trainer
        confound_attr=args.confound_attr
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Run evaluation only or full training
    if args.eval_only:
        print("Running evaluation only...")
        trainer.test(epoch=0)
    else:
        trainer.train()