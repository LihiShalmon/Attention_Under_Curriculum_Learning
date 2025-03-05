import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

CIFAR100_SUPERCLASS_LABELS = {
    0: [4, 30, 55, 72, 95],
    1: [1, 32, 67, 73, 91],
    2: [54, 62, 70, 82, 92],
    3: [9, 10, 16, 28, 61],
    4: [0, 51, 53, 57, 83],
    5: [22, 39, 40, 86, 87],
    6: [5, 20, 25, 84, 94],
    7: [6, 7, 14, 18, 24],
    8: [3, 42, 43, 88, 97],
    9: [12, 17, 37, 68, 76],
    10: [23, 33, 49, 60, 71],
    11: [15, 19, 21, 31, 38],
    12: [34, 63, 64, 66, 75],
    13: [26, 45, 77, 79, 99],
    14: [2, 11, 35, 46, 98],
    15: [27, 29, 44, 78, 93],
    16: [36, 50, 65, 74, 80],
    17: [47, 52, 56, 59, 96],
    18: [8, 13, 48, 58, 90],
    19: [41, 69, 81, 85, 89]
}

class BaseDatasetLoader:
    """Base class for dataset preparation and loading."""
    
    def __init__(self, dataset_name, num_classes, transform_train, transform_test):
        self.dataset_name = f"{dataset_name}_{num_classes}classes"
        self.train_data_path = f"datasets/saved_data_files/{self.dataset_name}_train.pt"
        self.test_data_path = f"datasets/saved_data_files/{self.dataset_name}_test.pt"
        self.num_classes = num_classes
        self.transform_train = transform_train
        self.transform_test = transform_test
        torch.serialization.add_safe_globals([CIFAR100])
        if os.path.exists(self.train_data_path) and os.path.exists(self.test_data_path):
            print(f"Loaded dataset from {self.train_data_path} and {self.test_data_path}.")
            self.train_dataset = torch.load(self.train_data_path, weights_only=False)
            self.test_dataset = torch.load(self.test_data_path, weights_only=False)
        else:
            self._create_and_save_datasets()
    
    def _create_and_save_datasets(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

class CIFAR100Loader(BaseDatasetLoader):
    """Handles CIFAR-100 dataset with optional superclass filtering."""
    
    def __init__(self, superclass_idx=None):
        self.superclass_idx = superclass_idx
        transform_train, transform_test = self._get_transforms()

        if superclass_idx is not None and superclass_idx not in CIFAR100_SUPERCLASS_LABELS:
            raise ValueError(f"Superclass index {superclass_idx} is not defined in CIFAR100_SUPERCLASS_LABELS")

        num_classes = len(CIFAR100_SUPERCLASS_LABELS.get(superclass_idx, [])) if superclass_idx is not None else 100
        super().__init__(f"cifar100_subset{superclass_idx}" if superclass_idx is not None else "cifar100", num_classes, transform_train, transform_test)
    
    def _get_transforms(self):
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return transform_train, transform_test

    def _create_and_save_datasets(self):
        # Load the full CIFAR-100 dataset
        train_dataset = CIFAR100(root="data", train=True, download=True, transform=self.transform_train)
        test_dataset = CIFAR100(root="data", train=False, download=True, transform=self.transform_test)
        
        # Filter by superclass if specified
        if self.superclass_idx is not None:
            print(f"Filtering dataset for superclass {self.superclass_idx}")
            train_dataset = self._filter_superclass(train_dataset)
            test_dataset = self._filter_superclass(test_dataset)
        
        # Assign datasets
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        
        # Save to disk
        os.makedirs("datasets/saved_data_files", exist_ok=True)
        torch.save(self.train_dataset, self.train_data_path)
        torch.save(self.test_dataset, self.test_data_path)
        print(f"Saved datasets to {self.train_data_path} and {self.test_data_path}")

    def _filter_superclass(self, dataset):
        superclass_mapping = self._get_superclass_mapping()
        selected_indices = [i for i, (_, target) in enumerate(dataset) 
                            if superclass_mapping.get(target) == self.superclass_idx]
        
        fine_labels = CIFAR100_SUPERCLASS_LABELS[self.superclass_idx]
        label_map = {label: idx for idx, label in enumerate(fine_labels)}
        
        filtered_data = []
        for idx in selected_indices:
            img, original_label = dataset[idx]
            mapped_label = label_map[original_label]
            filtered_data.append((img, mapped_label))
        
        return filtered_data
    
    @staticmethod
    def _get_superclass_mapping():
        mapping = {}
        for superclass, sublist in CIFAR100_SUPERCLASS_LABELS.items():
            for i in sublist:
                mapping[i] = superclass
        return mapping

class PrepareData:
    """Class to manage dataset loading for multiple datasets."""
    
    def __init__(self, dataset_name, superclass_idx=None):
        self.dataset_name = dataset_name.lower()
        self.superclass_idx = superclass_idx
        self.dataset_loader = self._initialize_loader()
        # self.dataset_loader = CIFAR100Loader(dataset_name, superclass_idx)
    
    
    def _initialize_loader(self):
        if self.dataset_name == "cifar100" or self.dataset_name.startswith("cifar100_subset") or self.dataset_name == ("cifar100_subset_16"):
            return CIFAR100Loader(superclass_idx=self.superclass_idx)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def get_train_dataset(self):
        return self.dataset_loader.get_train_dataset()
    
    def get_test_dataset(self):
        return self.dataset_loader.get_test_dataset()

if __name__ == '__main__':
    try:
        dataset = PrepareData(dataset_name="cifar100_subset_16", superclass_idx=16)
        train_data = dataset.get_train_dataset()  # Call the method with ()
        test_data = dataset.get_test_dataset()    # Call the method with ()
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")
    except ValueError as e:
        print(f"Error: {e}")