import tensorflow as tf
import os

# Inisialisasi layer dan utility
layers = tf.keras.layers
models = tf.keras.models
image_dataset_from_directory = tf.keras.utils.image_dataset_from_directory

# 1. Konfigurasi Dataset (Menggunakan struktur yang sama [cite: 25])
base_dir = 'dataset/dataset_terbaru/Dataset_Alphabet_Nomor'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

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

# 2. Data Augmentation (Teknik yang terbukti efektif meningkatkan akurasi)
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.05, 0.05),
])

# 3. Membangun Model MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(64, 64, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False 

model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    data_augmentation,
    layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

# 4. Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Callbacks (Early Stopping untuk menjaga model tetap Well-Fit [cite: 512, 650])
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 6. Training Process
history = model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,
    callbacks=[early_stop]
)

# 7. Evaluasi dan Penyimpanan
print("\n--- Evaluasi Data Test (MobileNetV2) ---")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Akurasi Test MobileNetV2: {test_acc*100:.2f}%")

os.makedirs('hasil/hasil_mobilenet', exist_ok=True)
model.save('hasil/hasil_mobilenet/MobileNetV2_Alphabet_Besar.keras')

print("Model MobileNetV2 berhasil disimpan untuk tahap komparasi arsitektur.")