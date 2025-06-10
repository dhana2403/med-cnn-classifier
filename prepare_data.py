
from utils.dataset_utils import prepare_binary_labels, handle_missing_values
from sklearn.model_selection import train_test_split

csv_path = "/Users/dhanalakshmijothi/Desktop/python/med_cnn_classifier/data/HAM10000_metadata.csv"
df = prepare_binary_labels(csv_path)
df = handle_missing_values(csv_path)

train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['binary_label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['binary_label'], random_state=42)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)
