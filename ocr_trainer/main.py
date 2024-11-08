import csv
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torchvision import transforms
from PIL import Image
from model import OCRModel

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    images = []
    labels = []

    with open('./data/labels.csv', 'r') as file:
        reader = csv.reader(file, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            label = row[0]
            image_path = row[1]

            image = Image.open(image_path).convert("L").resize((50, 50))
            image = np.array(image, dtype=np.float32) / 255.0
            images.append(image)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    label_mapping = list(label_encoder.classes_)

    with open('label_mapping.json', 'w') as json_file:
        json.dump(label_mapping, json_file)

    x_train, x_temp, y_train, y_temp = train_test_split(images, encoded_labels, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=1/3, random_state=42)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(lambda img: img.convert('L')),
        transforms.RandomRotation(3),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(50, scale=(0.9, 1.1)),
        transforms.ToTensor()
    ])

    class OCRDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            else:
                image = torch.tensor(image, dtype=torch.float32)
                image = np.expand_dims(image, axis=0)

            return image, label

    train_dataset = OCRDataset(x_train, y_train, transform=train_transform)
    val_dataset = OCRDataset(x_val, y_val)
    test_dataset = OCRDataset(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = OCRModel(num_classes=len(set(labels))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, "
            f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {100 * correct / total:.2f}%")

    # Save the model
    torch.save(model.state_dict(), 'model_ocr_weights.pth')
