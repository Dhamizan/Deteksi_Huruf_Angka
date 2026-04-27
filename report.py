import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import keras
from keras.utils import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# 1. Konfigurasi Path
model_path = 'hasil/hasil_dataset_baru/CNN_Dataset_Huruf_Besar.keras'
log_path = 'hasil/log/training_log_Besar.csv' # Path file CSV kamu
base_dir = 'dataset/dataset_terbaru/Dataset_Alphabet_Besar'
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

# Load Model & Log
print("Memuat model dan log training...")
model = keras.models.load_model(model_path, compile=False)
history_df = pd.read_csv(log_path)

# 2. Load Dataset (Untuk Confusion Matrix)
test_ds = image_dataset_from_directory(
    os.path.join(base_dir, 'test'),
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)
class_names = test_ds.class_names

def get_predictions(dataset):
    y_true, y_pred = [], []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    return np.array(y_true), np.array(y_pred)

# 3. Eksekusi Evaluasi
print("--- Mengevaluasi Data Testing ---")
y_true_test, y_pred_test = get_predictions(test_ds)
test_acc_final = np.mean(y_true_test == y_pred_test)

# Ambil akurasi terakhir dari CSV untuk Analisis Fitting
train_acc_csv = history_df['accuracy'].iloc[-1]
val_acc_csv = history_df['val_accuracy'].iloc[-1]
gap = abs(train_acc_csv - val_acc_csv)

if train_acc_csv < 0.60:
    status = "UNDERFITTING"
elif gap > 0.10:
    status = "OVERFITTING"
elif gap <= 0.05:
    status = "WELL-FIT"
else:
    status = "GOOD-FIT"

print("\n" + "="*35)
print(f"   HASIL ANALISIS FITTING: {status}")
print("="*35)
print(f"Akurasi Training (Log): {train_acc_csv*100:.2f}%")
print(f"Akurasi Testing  (Log): {val_acc_csv*100:.2f}%")
print(f"Gap Akurasi           : {gap*100:.2f}%")
print("="*35)

# --- VISUALISASI 1: GRAFIK LOSS & ACCURACY DARI CSV ---
plt.figure(figsize=(14, 5))

# Plot Loss (Sesuai gambar referensimu)
plt.subplot(1, 2, 1)
plt.plot(history_df['epoch'], history_df['loss'], label='Training Loss', color='#1f77b4', linewidth=2)
plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2)
plt.title('Visualize the Loss of Both Train and Test', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Plot Accuracy (Sesuai permintaan: Oren Train, Biru Test)
plt.subplot(1, 2, 2)
plt.plot(history_df['epoch'], history_df['accuracy'] * 100, label='Training Accuracy', color='#ff7f0e', linewidth=2)
plt.plot(history_df['epoch'], history_df['val_accuracy'] * 100, label='Test Accuracy', color='#1f77b4', linewidth=2)
plt.title(f'Accuracy Analysis (Status: {status})', fontsize=12, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 105)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# --- VISUALISASI 2: CONFUSION MATRIX ---
cm = confusion_matrix(y_true_test, y_pred_test)
plt.figure(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title(f"Confusion Matrix (Final Test Acc: {test_acc_final*100:.2f}%)")
plt.show()

# --- VISUALISASI 3: HEATMAP REPORT ---
report_dict = classification_report(y_true_test, y_pred_test, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).T 

plt.figure(figsize=(10, 8))
sns.heatmap(report_df, annot=True, fmt='.4f', cmap='Blues', cbar=True)
plt.title("Classification Report Heatmap")
plt.show()

print("\n=== Detail Classification Report ===")
print(classification_report(y_true_test, y_pred_test, target_names=class_names, digits=4))