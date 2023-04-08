"""To train a plant disease detector."""

import torch
import logging
from torchvision import datasets

from model.cnn import CNN_Model
from train.fit import batch_gd
from utils.load_args import get_args
from utils.data import data_transforms, split_dataset, dataloader

logging.basicConfig(
    filename="runing.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
)

if __name__ == "__main__":
    args = get_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        dataset = datasets.ImageFolder(args.data_path, transform=data_transforms())
    except FileNotFoundError:
        logging.error(f"No such file or directory: {args.data_path}")
    train_sampler, validation_sampler, test_sampler = split_dataset(dataset, args.train_per, args.val_per, args.test_per)

    train_loader, validation_loader, test_loader = dataloader(dataset, args.batchsize, train_sampler, validation_sampler, test_sampler)

    model = CNN_Model(len(dataset.class_to_idx))
    
    logging.info("Start training...")
    batch_gd(model=model, 
             train_loader=train_loader, 
             validation_loader=validation_loader, 
             epochs=args.epochs, 
             lr=args.lr,
             weights_path=args.wgts_path,
             device=DEVICE)
    logging.info("Finished!")