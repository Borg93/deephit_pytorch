import torch
from torch.utils.data import DataLoader, Dataset


class CompetingRiskDataset(Dataset):
    """Competing risk dataset."""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Remove unwanted columns from the item dictionary
        columns_to_remove = [
            "__index_level_0__",
            "time",
            "label",
            "true_time",
            "true_label",
            "quantile",
        ]
        sample = torch.tensor([value for key, value in item.items() if key not in columns_to_remove])

        label = torch.tensor([(item["label"], item["quantile"])])  # Get the 'quantile' value as the label

        return sample, label


if __name__ == "__main__":
    from preprocess import preprocess_pipe

    dataset_transformed_train, dataset_transformed_val, dataset_transformed_test = preprocess_pipe(
        dataset_hf="Gabriel/synthetic_competing_risk"
    )

    training_data = CompetingRiskDataset(dataset_transformed_train)
    validation_data = CompetingRiskDataset(dataset_transformed_val)
    test_data = CompetingRiskDataset(dataset_transformed_test)

    training_loader = DataLoader(training_data, batch_size=256, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=True)
