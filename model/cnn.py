import logging
from torch import nn

class CNN_Model(nn.Module):

    def __init__(self, K):
        super(CNN_Model, self).__init__()
        self.conv1 = self._block(c1_in=3, c2_in=32, c_out=32, b_norm=32)
        self.conv2 = self._block(c1_in=32, c2_in=64, c_out=64, b_norm=64)
        self.conv3 = self._block(c1_in=64, c2_in=128, c_out=128, b_norm=128)
        self.conv4 = self._block(c1_in=128, c2_in=256, c_out=256, b_norm=256)

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def _block(self, c1_in, c2_in, c_out, b_norm, k=3, p=1, pk=2):
        return nn.Sequential(
            nn.Conv2d(in_channels=c1_in, out_channels=c_out, kernel_size=k, padding=p),
            nn.ReLU(),
            nn.BatchNorm2d(b_norm),
            nn.Conv2d(in_channels=c2_in, out_channels=c_out, kernel_size=k, padding=p),
            nn.ReLU(),
            nn.BatchNorm2d(b_norm),
            nn.MaxPool2d(pk),
        )

    def forward(self, X):
        out = self.conv1(X)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        # Flatten
        out = out.view(-1, 50176)

        # Fully connected
        out = self.dense_layers(out)
        return out
    
if __name__ == "__main__":
    TARGETS_SIZE = 39
    cnn_model = CNN_Model(TARGETS_SIZE)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(name)s-%(levelname)s-%(message)s")
    logging.info(f"CNN_Model: {cnn_model}")
