import torch
from torchvision import transforms
from utils.dataset_utils import HAM10000Dataset
from models.cnn_model import CNN
from torch.utils.data import DataLoader
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

test_df = pd.read_csv("test.csv")
image_dir = "/Users/dhanalakshmijothi/Desktop/python/med_cnn_classifier/data/images"

test_dataset = HAM10000Dataset(test_df, image_dir, transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = CNN()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images= images.to(device)
        outputs=model(images)
        _, preds= torch.max(outputs,1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

        print("Test Accuracy:", accuracy_score(all_labels, all_preds))
        print("Classification Report:\n", classification_report(all_labels, all_preds))
        print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))