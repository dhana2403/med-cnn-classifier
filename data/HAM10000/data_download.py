import os 
import zipfile

def download_ham10000(data_dir="data/HAM10000"):
    metadata_path = os.path.join(data_dir, "HAM10000_metadata.csv")
    images_dir = os.path.join(data_dir, "images")
    images_path1 = os.path.join(data_dir, "HAM10000_images_part_1")
    images_path2 = os.path.join(data_dir, "HAM10000_images_part_2")

    zip_path1 = os.path.join(data_dir, "HAM10000_images_part_1.zip")
    zip_path2 = os.path.join(data_dir, "HAM10000_images_part_2.zip")

    os.makedirs(images_dir, exist_ok=True)

    if os.path.exists(metadata_path) and os.path.exists(images_path1) and os.path.exists(images_path2):
        print("HAM10000 dataset is present.")
        return
    
    if os.path.exists(zip_path1) and not os.path.exists(images_path1):
        print(f"Unzipping {zip_path1}...")
        with zipfile.ZipFile(zip_path1, 'r') as zip_ref:
            zip_ref.extractall(images_dir)

    if os.path.exists(zip_path2) and not os.path.exists(images_path2):
        print(f"Unzipping {zip_path2}...")
        with zipfile.ZipFile(zip_path2, 'r') as zip_ref:
            zip_ref.extractall(images_dir)

    if not os.path.exists(metadata_path) and os.path.exists(images_path1) and os.path.exists(images_path2):
        print("Please download the HAM10000 dataset manually from Kaggle:")
        print("https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print(f"Then place the files and folders in: {os.path.abspath(data_dir)}")
    else: 
        print("HAM10000 dataset is present after extraction")

if __name__ == '__main__':
    download_ham10000()
