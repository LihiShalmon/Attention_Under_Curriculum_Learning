from b_prepare_curriculum import PrepareCurriculum
from a_prepare_data import PrepareData
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import random
import pandas as pd
import os
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CurriculumScheduler:
    def __init__(self, samples_by_difficulty, total_epochs, pacing_function_type="linear", start_fraction=0.3, inc=2.0, step_length=1):
        self.samples_by_difficulty = samples_by_difficulty
        self.total_epochs = total_epochs
        self.pacing_function_type = pacing_function_type
        self.start_fraction = start_fraction
        self.inc = inc
        self.step_length = step_length
        self.total_samples = len(samples_by_difficulty)
        self.cur_num_samples = 0
        self.end_fraction = 1.0  # Use all samples at the end of training

    def get_indices_for_epoch(self, epoch):
        if self.pacing_function_type == "exponential":
            # Exponential pacing: fraction = start_fraction * (inc ** ((epoch - 1)/step_length))
            fraction = self.start_fraction * (self.inc ** ((epoch - 1) / self.step_length))
            fraction = min(fraction, 1.0)
        elif self.pacing_function_type == "linear":
            if self.total_epochs <= 1:
                fraction = 1.0
            else:
                fraction = self.start_fraction + (1.0 - self.start_fraction) * ((epoch - 1) / (self.total_epochs - 1))
        else:
            raise ValueError("pacing_function_type must be 'linear' or 'exponential'")
        num_samples = int(fraction * self.total_samples)
        num_samples = max(num_samples, 1)
        self.cur_num_samples = num_samples
        return self.samples_by_difficulty[:num_samples]

def validate_curriculum_scheduler(curriculum_scheduler, epochs=10):
    """Quick validation to check increasing sample sizes."""
    for epoch in range(1, epochs + 1):
        indices = curriculum_scheduler.get_indices_for_epoch(epoch)
        fraction = len(indices) / curriculum_scheduler.total_samples
        print(f"Epoch {epoch}: Using {len(indices)} samples ({fraction:.2f} of dataset)")

def train_model(model, train_loader, optimizer, criterion, device):
    """Simple training function for one epoch."""
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

if __name__ == '__main__':
    from torchvision.models import resnet18
    # Configuration for testing
    dataset = "cifar100_subset_mammals"
    superclass_idx = 16
    total_epochs = 2
    batch_size = 128

    prepare_data = PrepareData(dataset, superclass_idx=superclass_idx)
    train_dataset = prepare_data.get_train_dataset()
    test_dataset = prepare_data.get_test_dataset()

    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    curriculum = PrepareCurriculum(model, dataset, superclass_idx=superclass_idx)
    scheduler = CurriculumScheduler(
        samples_by_difficulty=curriculum.samples_by_difficulty,
        total_epochs=total_epochs
    )

    print("Validating curriculum scheduler:")
    validate_curriculum_scheduler(scheduler, epochs=10)

    print("\nStarting training...")
    for epoch in range(total_epochs):
        current_indices = scheduler.get_indices_for_epoch(epoch)
        sampler = SubsetRandomSampler(current_indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        avg_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{total_epochs}, Loss: {avg_loss:.4f}, Samples: {len(current_indices)}")
