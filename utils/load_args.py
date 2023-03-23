import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./plant_leave_diseases_dataset_with_augmentation")

    parser.add_argument('--train_per', type=float, default=0.6)
    parser.add_argument('--val_per', type=float, default=0.25)
    parser.add_argument('--test_per', type=float, default=0.15)

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wgts_path', type=str, default="weights/plant_disease_model_latest_py.pt")
    args = parser.parse_args()
    return args