import time
from collections import Counter
import os
import wandb
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim.lr_scheduler as lr_scheduler
from a_prepare_data import PrepareData
from b_prepare_curriculum import PrepareCurriculum
from c_prepare_schedual import CurriculumScheduler
from models.cifar10resnet import Cifar10ResNet
from models.original_basic_block import OriginalBasicBlock


class IndexedDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        if isinstance(dataset, list):
            # Extract targets from list of (img, label) tuples
            self.targets = dataset
        elif hasattr(dataset, 'targets'):
            self.targets = dataset.targets
        elif isinstance(dataset, dict) and 'targets' in dataset:
            self.targets = dataset['targets']
        else:
            raise AttributeError(f"Dataset does not provide 'targets'. Type: {type(dataset)}")

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


def get_TRANSFER_LEARNING_MODEL():
    model = Cifar10ResNet(OriginalBasicBlock, [3, 3, 3], num_classes=5)
    checkpoint_path = "models/saved_models/baseline_resnet20_5_epochs.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint if "model_state_dict" not in checkpoint else checkpoint["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Pretrained checkpoint not found at {checkpoint_path}")
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RunExp:
    def __init__(
            self,
            dataset,
            curriculum="vanilla",
            output_path="models/saved_models/",
            batch_size=128,
            num_epochs=5,
            learning_rate=0.1,
            lr_decay_rate=1.5,
            minimal_lr=1e-4,
            lr_batch_size=300,
            processor="cpu",
            lr_scheduler_type="ExponentialLR",
            scheduler_params=None,
            run_name=None,
            used_model=None,
            pacing_function="exponential",  # New
            start_fraction=0.3,  # New
            inc=2.0,  # New
            step_length=1,  # New
            num_classes=5,
            momentum=0.9,
            weight_decay=5e-4
    ):
        self.initialize_all_seeds(processor)

        # Experiment configurations
        self.transfer_learning_model = get_TRANSFER_LEARNING_MODEL()
        self.used_model = used_model
        self.num_classes = num_classes
        self.run_name = run_name
        self.dataset = dataset
        self.curriculum = curriculum
        self.output_path = output_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.minimal_lr = minimal_lr
        self.lr_batch_size = lr_batch_size
        self.lr_scheduler_type = lr_scheduler_type
        self.scheduler_params = scheduler_params if scheduler_params else {}
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.pacing_function = pacing_function
        self.start_fraction = start_fraction
        self.inc = inc
        self.step_length = step_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else processor)
        self.difficulty_csv_path = f"data/saved_difficulty_scores/{self.dataset}.csv"
        if os.path.exists(self.difficulty_csv_path):
            self.difficulty_df = pd.read_csv(self.difficulty_csv_path)
            self.loss_map = dict(zip(self.difficulty_df['SampleIndex'], self.difficulty_df['Loss']))
        else:
            raise FileNotFoundError(f"Difficulty scores CSV not found at {self.difficulty_csv_path}")

        # Initialize model, optimizer, and LR scheduler
        self.model = self.build_model()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.lr_scheduler = self.get_lr_scheduler()
        self.num_classes = num_classes

    def initialize_all_seeds(self, processor):
        seed = 7
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else processor)

    def initialize_wandb(self, curriculum_scheduler):
        wandb.init(
            project="CNN_With_Curriculum_Learning",
            name=self.generate_run_name(),
            config={
                "learning_rate": self.learning_rate,
                "momentum": self.momentum,
                "weight_decay": self.weight_decay,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "curriculum_learning": curriculum_scheduler is not None,
                "lr_scheduler_type": self.lr_scheduler_type,
                "pacing_function": self.pacing_function
            },
        )

    def get_lr_scheduler(self):
        if self.lr_scheduler_type == "MultiStepLR":
            return lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.scheduler_params.get("milestones", [100, 150]),
                gamma=self.scheduler_params.get("gamma", 0.1),
            )
        elif self.lr_scheduler_type == "ExponentialLR":
            return lr_scheduler.ExponentialLR(
                self.optimizer, gamma=self.scheduler_params.get("gamma", 0.95)
            )
        else:
            raise ValueError(f"Unknown LR scheduler type: {self.lr_scheduler_type}")

    def build_model(self):
        try:
            print(f"Constructing model for {self.generate_run_name()}")
            if self.used_model == 'resnet20_selfattn':
                from models.resnet_self_att import ResnetSelfAtt
                model = ResnetSelfAtt(OriginalBasicBlock, [3, 3, 3], num_classes=self.num_classes)
            elif self.used_model == 'resnet20_mhattn':
                from models.resnet_multi_head_att import ResnetMultiHeadAtt
                model = ResnetMultiHeadAtt(OriginalBasicBlock, [3, 3, 3], num_classes=self.num_classes, num_heads=4)
            elif self.used_model == 'resnet20_cbam':
                from models.resnet_cbam import ResnetCBAM
                model = ResnetCBAM(OriginalBasicBlock, [3, 3, 3], num_classes=self.num_classes)
            elif self.used_model == 'resnet20':
                model = Cifar10ResNet(OriginalBasicBlock, [3, 3, 3], num_classes=5)
            else:
                raise ValueError(f"Unknown model type: {self.used_model}")
            return model.to(self.device)
        except Exception as e:
            print(f"Error in build_model: {e}")
        raise

    def generate_run_name(self):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if hasattr(self, 'run_name') and self.run_name is not None:
            self.run_name = f"{self.run_name}_{timestamp}"
            return self.run_name
        else:
            self.run_name = (f"{self.dataset}_{self.curriculum}_{self.lr_scheduler_type}_{timestamp}")
            return self.run_name

    def run_experiment(self):
        # Prepare data
        data_prep = PrepareData(dataset_name=self.dataset, superclass_idx=16)
        train_dataset = IndexedDatasetWrapper(data_prep.get_train_dataset())
        test_dataset = IndexedDatasetWrapper(data_prep.get_test_dataset())

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Randomize for vanilla
            num_workers=2,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # Initialize curriculum scheduler based on curriculum type
        curriculum_scheduler = None
        sorted_indices = None  # Initialize to None by default
        if self.curriculum == "curriculum":
            curriculum_maker = PrepareCurriculum(
                used_model=self.transfer_learning_model,
                used_dataset_name=self.dataset,
                total_epochs=self.num_epochs,
                device=self.device
            )
            sorted_indices = curriculum_maker.balance_order(curriculum_maker.samples_by_difficulty, train_dataset)
            print("Curriculum mode (increasing difficulty)")
        elif self.curriculum == "anti-curriculum":
            curriculum_maker = PrepareCurriculum(
                used_model=self.transfer_learning_model,
                used_dataset_name=self.dataset,
                total_epochs=self.num_epochs,
                device=self.device
            )
            sorted_indices = curriculum_maker.balance_order(curriculum_maker.samples_by_difficulty[::-1], train_dataset)
            print("Anti-Curriculum mode (decreasing difficulty)")
        else:
            print("Vanilla mode (random sampling)")

        if self.curriculum != "vanilla":
            curriculum_scheduler = CurriculumScheduler(
                samples_by_difficulty=sorted_indices,
                total_epochs=self.num_epochs,
                pacing_function_type=self.pacing_function,
                start_fraction=self.start_fraction,
                inc=self.inc,
                step_length=self.step_length
            )
            self.schedualer = curriculum_scheduler
            print("First 5 samples:", sorted_indices[:5])
            print("Last 5 samples:", sorted_indices[-5:])
            print("Losses for first 5:", [self.loss_map.get(idx, 0) for idx in sorted_indices[:5]])
            print("Losses for last 5:", [self.loss_map.get(idx, 0) for idx in sorted_indices[-5:]])

        # In vanilla mode, sorted_indices is None, so pass that (or simply pass curriculum_scheduler)
        self.train_model(train_loader, test_loader, sorted_indices, curriculum_scheduler)

    def train_model(self, train_loader, test_loader, sorted_indices, curriculum_scheduler=None):
        self.initialize_wandb(curriculum_scheduler)
        print(f"Started training {self.generate_run_name()}")

        for epoch in range(1, self.num_epochs + 1):
            train_loader_curriculum = self.apply_training_curriculum(train_loader, curriculum_scheduler, epoch)

            if curriculum_scheduler:
                subset = train_loader_curriculum.dataset
                num_samples = len(subset)
                indices = subset.indices
                losses = [self.loss_map.get(int(idx), 0) for idx in indices]
                avg_difficulty = sum(losses) / num_samples if num_samples > 0 else 0
            else:
                num_samples = len(train_loader.dataset)
                avg_difficulty = None

            train_loss, train_accuracy = self.train_one_epoch(train_loader_curriculum, epoch)
            test_loss, test_accuracy = self.evaluate_model(test_loader, epoch)

            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(test_loss)
            else:
                self.lr_scheduler.step()

            self.log_wandb(epoch, train_loss, train_accuracy, test_loss, test_accuracy, num_samples, avg_difficulty)

            if curriculum_scheduler:
                print(
                    f"Epoch {epoch}: train_loss {train_loss:.4f}, test_loss {test_loss:.4f}, test_accuracy {test_accuracy:.2f}%, num_samples {num_samples}, avg_difficulty {avg_difficulty:.4f}")
            else:
                print(
                    f"Epoch {epoch}: train_loss {train_loss:.4f}, test_loss {test_loss:.4f}, test_accuracy {test_accuracy:.2f}%, num_samples {num_samples}")

            self.log_precomputed_metrics(epoch, train_loader_curriculum)

        self.save_model()
        wandb.finish()

    def apply_training_curriculum(self, train_loader, curriculum_scheduler, epoch):
        if curriculum_scheduler:
            current_indices = curriculum_scheduler.get_indices_for_epoch(epoch)
            subset = torch.utils.data.Subset(train_loader.dataset, current_indices)
            return torch.utils.data.DataLoader(subset, batch_size=train_loader.batch_size, shuffle=True)
        return train_loader

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        criterion = nn.CrossEntropyLoss()

        for inputs, targets, _ in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = running_loss / max(1, len(loader))
        accuracy = 100.0 * correct / max(1, total)
        print(f"Epoch {epoch:03d}: Train Loss = {avg_loss:.4f}, Train Acc = {accuracy:.2f}%")
        return avg_loss, accuracy

    def evaluate_model(self, loader, epoch):
        self.model.eval()
        test_loss, correct, total = 0.0, 0, 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets, _ in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = test_loss / len(loader)
        accuracy = 100.0 * correct / total
        print(f"Epoch {epoch:03d}: Test Loss = {avg_loss:.4f}, Test Acc = {accuracy:.2f}%")
        return avg_loss, accuracy

    def save_model(self):
        save_dir = os.path.join("models", "saved_models")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.generate_run_name()}.pth")

        torch.save(
            {
                "epoch": self.num_epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.lr_scheduler.state_dict(),
                "learning_rate": self.learning_rate,
            },
            save_path,
        )
        print(f"✅ Model saved to {save_path}")

    def log_wandb(self, epoch, train_loss, train_acc, test_loss, test_acc, num_samples, avg_difficulty):
        log_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "num_samples": num_samples
        }
        if avg_difficulty is not None:
            log_dict["avg_difficulty"] = avg_difficulty
        wandb.log(log_dict)

    def log_precomputed_metrics(self, epoch, loader):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets, indices) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_loss_values = [self.loss_map.get(int(idx.item()), 0) for idx in indices]
                batch_loss = torch.tensor(batch_loss_values, device=self.device).mean()
                from collections import Counter
                label_counter = Counter(targets.cpu().numpy())
                total_samples = sum(label_counter.values())
                normalized_label_counts = {int(label): count / total_samples for label, count in label_counter.items()}
                wandb.log({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "batch_precomputed_loss": batch_loss.item(),
                    "normalized_label_counts_per_batch": normalized_label_counts
                })
                print(f"Epoch {epoch} - Batch {batch_idx}: Avg Precomputed Loss = {batch_loss:.4f}")


if __name__ == "__main__":
    # vanilla, curriculum , anti
    # resnet, resnet-spatial, resnet-channel, resnet-cbam
    ### 12
    #
    # # Train with curriculum learning
    curriculum_exp = RunExp(
        dataset="cifar100_subset_16",
        curriculum="curriculum",
        run_name="curriculum_resnet20",
        used_model="resnet20",
        num_epochs=50,
        learning_rate=0.085,
        lr_decay_rate=0.9,
        weight_decay=0.007
    )
    curriculum_exp.run_experiment()
    #
    # # Train without curriculum learning (vanilla training)
    # vanilla_exp = RunExp(
    #     dataset="cifar100_subset_16",
    #     curriculum="vanilla",
    #     run_name="baseline_resnet20",
    #     used_model="resnet20",
    #     num_epochs=100,
    #     learning_rate=0.085,
    #     lr_decay_rate=0.9,
    #     weight_decay=0.007
    # )
    # vanilla_exp.run_experiment()
    #
    #
    # # Train Resnet with self attention - vanilla
    # resnet_selfattn = RunExp(
    #     dataset="cifar100_subset_16",
    #     curriculum="vanilla",
    #     run_name="selfattn_vanilla_resnet20",
    #     used_model="resnet20_selfattn",
    #     num_epochs=100,
    #     learning_rate=0.01,
    #     lr_decay_rate=0.9,
    #     weight_decay=0.007
    # )
    # resnet_selfattn.run_experiment()
    #
    # # Train Resnet with self attention - curriculum
    # resnet_selfattn = RunExp(
    #     dataset="cifar100_subset_16",
    #     curriculum="curriculum",
    #     run_name="selfattn_curriculum_resnet20",
    #     used_model="resnet20_selfattn",
    #     num_epochs=5,
    #     learning_rate=0.01,
    #     lr_decay_rate=0.9,
    #     weight_decay=0.007
    # )
    # resnet_selfattn.run_experiment()
    #
    # # Train Resnet with multihead attention - vanilla
    # resnet_mhattn = RunExp(
    #     dataset="cifar100_subset_16",
    #     curriculum="vanilla",
    #     run_name="resnet20_vanilla_mhattn",
    #     used_model="resnet20_mhattn",
    #     num_epochs=2,
    #     learning_rate=0.01
    # )
    # resnet_mhattn.run_experiment()
    #
    # # Train Resnet with CBAM - vanilla
    # resnet_cbam = RunExp(
    #     dataset="cifar100_subset_16",
    #     curriculum="vanilla",
    #     run_name="resnet20_vanilla_cbam",
    #     used_model="resnet20_cbam",
    #     num_epochs=2,
    #     learning_rate=0.01
    # )
    # resnet_cbam.run_experiment()
    #
    # # Train Resnet with CBAM - curriculum
    # resnet_cbam = RunExp(
    #     dataset="cifar100_subset_16",
    #     curriculum="curriculum",
    #     run_name="resnet20_curriculum_cbam",
    #     used_model="resnet20_cbam",
    #     num_epochs=2,
    #     learning_rate=0.01
    # )
    # resnet_cbam.run_experiment()
