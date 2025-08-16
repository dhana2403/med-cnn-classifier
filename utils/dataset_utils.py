import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.metadata = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_id'] + '.jpg')
        image = Image.open(image_path).convert('RGB')

        if self.transform:
          image = self.transform(image)
          label = row['binary_label']
          return image, label

def prepare_binary_labels(csv_path):
    df = pd.read_csv(csv_path)
    benign_labels = ['nv', 'bkl', 'df', 'vasc']
    malignant_labels = ['akiec', 'bcc', 'mel']
    label_map = {label: 0 for label in benign_labels}
    label_map.update({label: 1 for label in malignant_labels})
    df['binary_label'] = df['dx'].map(label_map)
    return df

def handle_missing_values(csv_path): #this step is not needed if you are not gonna use metadata
    df = pd.read_csv(csv_path)
    df['age'] = df['age'].fillna(df['age'].median())
    return df
