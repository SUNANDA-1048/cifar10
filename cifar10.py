import argparse
import json
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

print("Environment variables:", os.environ)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# SimpleResNet18 model definition (replace with your actual model)
class SimpleResNet18(nn.Module):
    
    def __init__(self, num_classes=10):
        super(SimpleResNet18, self).__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # First block (two convolution layers + max-pooling)
        self.block1 = self._make_block(64, 64, 2)

        # Second block (two convolution layers + max-pooling)
        self.block2 = self._make_block(64, 128, 2)

        # Third block (one convolution layer)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Fourth block (one convolution layer)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.block2(x)

        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

    def _make_block(self, in_channels, out_channels, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
    # ... (same as your original definition)

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleResNet18()
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location=device))
    return model.to(device)


def _train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device Type: {}".format(device))

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )

    logger.info("Model loaded")
    model = SimpleResNet18()

    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(args.epochs):
        running_loss = 0.0
        model.train()  # Set the model to training mode
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        # Set the model to evaluation mode after each epoch
        model.eval()

    print("Finished Training")
    return _save_model(model, args.model_dir)

def _save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2, metavar="W",
                        help="number of data loading workers (default: 2)")
    parser.add_argument("--epochs", type=int, default=2, metavar="E",
                        help="number of total epochs to run (default: 2)")
    parser.add_argument("--batch-size", type=int, default=4, metavar="BS",
                        help="batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR",
                        help="initial learning rate (default: 0.001)")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M",
                        help="momentum (default: 0.9)")
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/default/path"))

    
    args = parser.parse_args()
    _train(args)
