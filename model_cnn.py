import tensorflow as tf
import os
import pandas as pd # Tambahkan ini untuk handle log
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

# 2. Model & Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.05, 0.05),
])

model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
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

# 3. CALLBACKS
# Early Stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# CSV Logger: Menyimpan angka loss/acc ke file agar tidak hilang
log_path = 'hasil/log/training_log_Besar.csv'
csv_logger = tf.keras.callbacks.CSVLogger(log_path, append=False)

# 4. TRAINING
history = model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,
    callbacks=[early_stop, csv_logger]
)

# 5. EVALUASI
print("\n Evaluasi Data Test")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Akurasi Test: {test_acc*100:.2f}%")

# 6. SIMPAN MODEL
model.save('hasil/hasil_dataset_baru/CNN_Dataset_Huruf_Besar.keras')
print(f"Log tersimpan di: {log_path}")

# 7. VISUALISASI ANALISIS FITTING
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

# Plot 1: Loss (Sesuai gambar yang kamu kirim)
plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, label='Training Loss', color='#1f77b4', linewidth=2)
plt.plot(epochs_range, val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2)
plt.title('Visualize the Loss of Both Train and Test')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

# Plot 2: Accuracy (Analisis Fitting)
plt.subplot(1, 2, 2)
plt.plot(epochs_range, acc, label='Training Accuracy', color='#ff7f0e', linewidth=2)
plt.plot(epochs_range, val_acc, label='Test Accuracy (Val)', color='#1f77b4', linewidth=2)
plt.title('Accuracy Analysis (Fitting)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()