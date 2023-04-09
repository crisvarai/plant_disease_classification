import logging
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

def data_transforms():
    return transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor() # Scale between [0, 1]
    ])

def split_dataset(dataset, train_per, val_per, test_per):
    sum = train_per + val_per + test_per
    assert sum == 1, f"expected that the sum of the values for splitting the dataset is equal to 1 , got {sum}"
    indices = list(range(len(dataset)))
    split = int(np.floor(0.85 * len(dataset)))
    validation = int(np.floor(0.70 * split))
    np.random.shuffle(indices)

    indices = list(range(len(dataset)))
    split = int(np.floor((1-test_per) * len(dataset)))
    validation = int(np.floor((1-test_per-val_per) * len(dataset)))
    np.random.shuffle(indices)

    train_ids, validation_ids, test_ids = (indices[:validation],
                                           indices[validation:split],
                                           indices[split:])
    
    train_sampler = SubsetRandomSampler(train_ids)
    validation_sampler = SubsetRandomSampler(validation_ids)
    test_sampler = SubsetRandomSampler(test_ids)
    return train_sampler, validation_sampler, test_sampler

def dataloader(dataset, batch_size, train_sampler, validation_sampler, test_sampler):
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    return train_loader, validation_loader, test_loader

    
if __name__ == "__main__":
    dataset = datasets.ImageFolder("../plant_leave_diseases_dataset_with_augmentation", transform=data_transforms())
    train_sampler, validation_sampler, test_sampler = split_dataset(dataset, 0.6, 0.25, 0.15)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(name)s-%(levelname)s-%(message)s")
    logging.info(f"Length of train size: {len(train_sampler)}")
    logging.info(f"Length of validation size: {len(validation_sampler)}")
    logging.info(f"Length of test size: {len(test_sampler)}")
    logging.info(f"Targets size: {len(dataset.class_to_idx)}")