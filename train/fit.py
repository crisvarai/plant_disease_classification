import torch
import numpy as np

from torch import nn
from tqdm import tqdm

def batch_gd(model, train_loader, validation_loader, epochs, lr, weights_path, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()  # this include softmax + cross entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = np.zeros(epochs)
    validation_losses = np.zeros(epochs)

    for e in range(epochs):
        train_loss = []
        model.train()
        bar = tqdm(train_loader)

        for inputs, targets in bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            train_loss.append(loss.item())  # torch to numpy world
            loss.backward()
            optimizer.step()
            bar.set_description(f"loss {np.mean(train_loss):.5f}")

        train_loss = np.mean(train_loss)
        validation_loss = []
        model.eval()
        bar = tqdm(validation_loader)

        with torch.no_grad():
            for inputs, targets in bar:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = criterion(output, targets)
                validation_loss.append(loss.item())  # torch to numpy world
                bar.set_description(f"val_loss {np.mean(validation_loss):.5f}")

            validation_loss = np.mean(validation_loss)
            train_losses[e] = train_loss
            validation_losses[e] = validation_loss
        print(f"Epoch:{e+1}/{epochs} Train_loss:{train_loss:.5f} Validation_loss:{validation_loss:.5f}")

    torch.save(model.state_dict(), weights_path)
    print("WEIGHTS-ARE-SAVED")
    return train_losses, validation_losses