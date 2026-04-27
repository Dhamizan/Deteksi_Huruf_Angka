import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt

# Inisialisasi
layers = tf.keras.layers
models = tf.keras.models
image_dataset_from_directory = tf.keras.utils.image_dataset_from_directory

base_dir = 'dataset/dataset_terbaru/Dataset_Alphabet_Besar'
IMG_SIZE = (64, 64)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

# 1. Load Dataset
train_ds = image_dataset_from_directory(
    os.path.join(base_dir, 'train'),
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    os.path.join(base_dir, 'valid'),
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = image_dataset_from_directory(
    os.path.join(base_dir, 'test'),
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

# 2. Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.05, 0.05),
])

# 3. Build MobileNetV2 Model
# Menggunakan base model tanpa top layer (lapisan klasifikasi akhir)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(64, 64, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model (agar tidak berubah di tahap awal training)
base_model.trainable = False 

model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    data_augmentation,
    # Preprocessing khusus MobileNetV2 (bukan 1./255)
    layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input),
    base_model,
    layers.GlobalAveragePooling2D(), # Pengganti Flatten agar lebih ringan
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# FOLDER HASIL
os.makedirs('hasil/log', exist_ok=True)
os.makedirs('hasil/hasil_mobilenet', exist_ok=True)

# 4. CALLBACKS
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Simpan log
log_path = 'hasil/log/training_log_mobilenet_Besar.csv'
csv_logger = tf.keras.callbacks.CSVLogger(log_path, append=False)

# 5. TRAINING
history = model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,
    callbacks=[early_stop, csv_logger]
)

# 6. EVALUASI
print("\n Evaluasi Data Test (MobileNetV2)")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Akurasi Test MobileNetV2: {test_acc*100:.2f}%")

# 7. SIMPAN MODEL
model.save('hasil/hasil_mobilenet/MobileNetV2_Dataset_Besar.keras')
print(f"Log tersimpan di: {log_path}")

# 8. VISUALISASI ANALISIS FITTING
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

# Plot 1: Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, label='Training Loss', color='#1f77b4', linewidth=2)
plt.plot(epochs_range, val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2)
plt.title('MobileNetV2: Loss Analysis')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

# Plot 2: Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, acc, label='Training Accuracy', color='#ff7f0e', linewidth=2)
plt.plot(epochs_range, val_acc, label='Test Accuracy (Val)', color='#1f77b4', linewidth=2)
plt.title('MobileNetV2: Accuracy Analysis')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()