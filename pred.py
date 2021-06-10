#! /usr/bin/python3

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import neuralnetwork

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model = neuralnetwork.CNN()
model.load_state_dict(torch.load("model.pth"))

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

model.eval()

correct = 0
all = 0

for X, y in test_data:
    with torch.no_grad():
        pred = model(X[None, ...])
        predicted, actual = classes[pred[0].argmax(0)], classes[y]

        all += 1

        if predicted == actual:
            correct += 1

print(f'Got correct: "{correct/all}"')
