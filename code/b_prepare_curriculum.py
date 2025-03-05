# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# import os
# from torch.utils.data import DataLoader
# from a_prepare_data import PrepareData

# class PrepareCurriculum:
#     def __init__(self, used_model, used_dataset_name, superclass_idx=None, total_epochs=200):
#         self.used_model = used_model
#         self.used_dataset_name = used_dataset_name
#         self.superclass_idx = superclass_idx
#         self.total_epochs = total_epochs
#         self.num_classes = 5  # Hardcoded for CIFAR-100 subset with 5 classes
#         self.saved_difficulty = f"data/saved_difficulty_scores/{used_dataset_name}.csv"
        
#         # if os.path.exists(self.saved_difficulty):
#         #     print(f"Loaded curriculum from {self.saved_difficulty}.")
#         #     self.samples_by_difficulty = pd.read_csv(self.saved_difficulty)['SampleIndex'].to_list()
#         #     self.num_classes =  len(pd.read_csv(self.saved_difficulty)['Target'].unique())
#         # else:
#         self.used_data = PrepareData(used_dataset_name, superclass_idx=self.superclass_idx)
#         self.num_classes = len(set(self.used_data.get_train_dataset().target))
#         train_dataset = self.used_data.get_train_dataset()
#         unique_labels = set([label for _, label in train_dataset])
#         print(f"Unique labels in train_dataset: {unique_labels}")
#         self.samples_by_difficulty, self.loss_values, self.target = self.compute_difficulty_order(
#             self.used_model, train_dataset, device="cpu"
#         )
#         df = pd.DataFrame({
#             'SampleIndex': self.samples_by_difficulty,
#             'Loss': self.loss_values,
#             'Target': self.targets  # New column for targets
#         })
#         os.makedirs("data/saved_difficulty_scores", exist_ok=True)
#         df.to_csv(self.saved_difficulty, index=False)
#         print(f"Saved difficulty scores with targets to {self.saved_difficulty}.")

#     def compute_difficulty_order(self, model, dataset, device, batch_size=128):
#         model.eval()
#         criterion = nn.CrossEntropyLoss(reduction='none')
#         data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#         loss_list = []
#         target_list = []

#         with torch.no_grad():
#             for inputs, targets in data_loader:
#                 print(f"Input shape: {inputs.shape}, Target shape: {target.shape}")
#                 inputs, targets = inputs.to(device), target.to(device)
#                 outputs = model(inputs)
#                 losses = criterion(outputs, target)
#                 loss_list.extend(losses.cpu().numpy().tolist())
#                 target_list.extend(target.cpu().numpy().tolist())  # Collect targets

#         # Sort by loss (smallest to largest) and align targets
#         sorted_indices = np.argsort(loss_list)
#         sorted_losses = np.array(loss_list)[sorted_indices]
#         sorted_targets = np.array(target_list)[sorted_indices]
#         return sorted_indices, sorted_losses, sorted_targets

#     def balance_order(self, order, dataset):
#         size_each_class = min([len([i for i in range(len(order)) if dataset.target[order[i]] == cls]) 
#                               for cls in range(self.num_classes)])
#         class_orders = []
#         for cls in range(self.num_classes):
#             class_orders.append([i for i in range(len(order)) if dataset.target[order[i]] == cls])
#         new_order = []
#         for group_idx in range(size_each_class):
#             group = sorted([class_orders[cls][group_idx] for cls in range(self.num_classes)])
#             for idx in group:
#                 new_order.append(order[idx])
#         return new_order

# if __name__ == '__main__':
#     from torchvision.models import resnet18

#     dataset_name = "cifar100_subset_16"
#     prepare_data = PrepareData(dataset_name=dataset_name, superclass_idx=16)
#     train_dataset = prepare_data.get_train_dataset()
#     test_dataset = prepare_data.get_test_dataset()

#     unique_labels = set(label for _, label in train_dataset)
#     print(f"Unique labels in train_dataset: {unique_labels}")
#     print(f"Train samples: {len(train_dataset)}")
#     print(f"Test samples: {len(test_dataset)}")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = resnet18(pretrained=False)
#     model.fc = nn.Linear(model.fc.in_features, 5)
#     model = model.to(device)

#     curriculum = PrepareCurriculum(model, dataset_name, superclass_idx=16)
#     sorted_indices = curriculum.balance_order(curriculum.samples_by_difficulty, train_dataset)
#     print(f"Balanced order first 10: {sorted_indices[:10]}")

#     # Verify saved CSV includes targets
#     df = pd.read_csv(curriculum.saved_difficulty)
#     print("Saved curriculum data (first 5 rows):")
#     print(df.head())
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from a_prepare_data import PrepareData


# Automatically use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrepareCurriculum:
    def __init__(self, used_model, used_dataset_name, device="cpu", superclass_idx=16, total_epochs=200):
        self.device = device
        self.used_model = used_model
        self.used_dataset_name = used_dataset_name
        self.superclass_idx = superclass_idx
        self.total_epochs = total_epochs

        # Load your (list-based) train data
        self.used_data = PrepareData(used_dataset_name, superclass_idx=self.superclass_idx)
        train_data = self.used_data.get_train_dataset()  # list of (img, label)
        # Optionally load twice if needed; otherwise, one call should suffice:
        train_data = self.used_data.get_train_dataset()
        labels_list = [label for _, label in train_data]
        self.labels_list = labels_list
        unique_labels = set(labels_list)
        self.num_classes = len(unique_labels)

        # File to save difficulty info
        self.saved_difficulty = f"data/saved_difficulty_scores/{used_dataset_name}.csv"

        # Compute difficulty order, passing the device
        self.samples_by_difficulty, self.loss_values, self.sorted_targets = \
            self.compute_difficulty_order(self.used_model, train_data, device=device)

        # Save results to CSV
        df = pd.DataFrame({
            'SampleIndex': self.samples_by_difficulty,
            'Loss': self.loss_values,
            'Target': self.sorted_targets
        })
        os.makedirs("data/saved_difficulty_scores", exist_ok=True)
        df.to_csv(self.saved_difficulty, index=False)
        print(f"Saved difficulty scores with targets to {self.saved_difficulty}.")

    def compute_difficulty_order(self, model, dataset, device, batch_size=128):
        """
        dataset is a list of (img, label) tuples.
        Creates a temporary DataLoader to run inference and compute losses.
        """
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction='none')
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        loss_list = []
        target_list = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                losses = criterion(outputs, targets)
                loss_list.extend(losses.cpu().numpy().tolist())
                target_list.extend(targets.cpu().numpy().tolist())

        sorted_indices = np.argsort(loss_list)
        sorted_losses = np.array(loss_list)[sorted_indices]
        sorted_targets = np.array(target_list)[sorted_indices]

        return sorted_indices, sorted_losses, sorted_targets

    def balance_order(self, order, dataset):
        """
        Balances the sorted samples so each class is equally represented.
        dataset is a list of (img, label) tuples.
        """
        size_each_class = min([
            len([i for i in range(len(order)) if dataset[order[i]][1] == cls])
            for cls in range(self.num_classes)
        ])

        class_orders = []
        for cls in range(self.num_classes):
            indices_for_cls = [i for i in range(len(order)) if dataset[order[i]][1] == cls]
            class_orders.append(indices_for_cls)

        new_order = []
        for group_idx in range(size_each_class):
            group = sorted([class_orders[cls][group_idx] for cls in range(self.num_classes)])
            for idx in group:
                new_order.append(order[idx])
        return new_order

# For testing purposes
if __name__ == '__main__':
    from torchvision.models import resnet18

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = "cifar100_subset_16"
    prepare_data = PrepareData(dataset_name=dataset_name, superclass_idx=16)
    train_dataset = prepare_data.get_train_dataset()
    test_dataset = prepare_data.get_test_dataset()

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.to(device)

    curriculum = PrepareCurriculum(model, dataset_name, device=device, superclass_idx=16)
    sorted_indices = curriculum.balance_order(curriculum.samples_by_difficulty, train_dataset)
    print(f"Balanced order first 10: {sorted_indices[:10]}")

    df = pd.read_csv(curriculum.saved_difficulty)
    print("Saved curriculum data (first 5 rows):")
    print(df.head())
