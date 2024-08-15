from torch import Tensor, nn, optim
import torch
import data_loader
from torchvision import transforms
from PIL import Image


class MelanomaCNN(nn.Module):
    def __init__(self):
        super(MelanomaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # input image size should be 128x128
        self.fc2 = nn.Linear(512, 1)  # Binary classification

    def forward(self, x: Tensor):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def train(train_csv, batch_size, learning_rate, num_epochs):
    train_loader = data_loader.create(train_csv, batch_size, shuffle=True)
    model = MelanomaCNN()
    model.train()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            labels: Tensor = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss: Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}"
        )

    torch.save(model.state_dict(), "melanoma_cnn.pth")
    print("Model training completed! Saved as melanoma_cnn.pth")


def evaluate(eval_csv, batch_size, model_path):
    eval_loader = data_loader.create(eval_csv, batch_size, shuffle=False)
    model = MelanomaCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    criterion = nn.BCELoss()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in eval_loader:
            labels: Tensor = labels.float().unsqueeze(1)
            outputs: Tensor = model(images)
            loss: Tensor = criterion(outputs, labels)
            val_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    val_loss /= len(eval_loader)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")


def inference(image_path, model_path):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image: Tensor = transform(image).unsqueeze(0)

    model = MelanomaCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        output = model(image)
        prediction = (output > 0.5).float().item()

    print(
        f"Prediction for {image_path}: {'Melanoma' if prediction == 1 else 'Not Melanoma'}"
    )
