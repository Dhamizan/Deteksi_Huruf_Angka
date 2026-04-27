import os
from PIL import Image
import random

INPUT_DIR = "dataset/dataset_raw/n"
#OUTPUT_DIR = "dataset/dataset_terbaru/Dataset_Alphabet_Kecil/test/n"
OUTPUT_DIR = "dataset/dataset_terbaru/Dataset_Alphabet_Kecil/train/n"
#OUTPUT_DIR = "dataset/dataset_terbaru/Dataset_Alphabet_Kecil/valid/n"
AG_PER_IMG = 29
IMG_SIZE = (1200, 900)

MAX_SHIFT = 150

os.makedirs(OUTPUT_DIR, exist_ok=True)

existing_files = [
    f for f in os.listdir(OUTPUT_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

count = len(existing_files)

def augment(img):
    img = img.resize(IMG_SIZE)

    angle = random.uniform(-16, 16)
    img = img.rotate(angle, fillcolor=255)
    
    shift_x = random.randint(-MAX_SHIFT, MAX_SHIFT)
    shift_y = random.randint(-MAX_SHIFT, MAX_SHIFT)

    img = img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, shift_x,
         0, 1, shift_y),
        fillcolor=255
    )

    return img

for file in os.listdir(INPUT_DIR):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(INPUT_DIR, file)
        img = Image.open(img_path).convert("L")

        for _ in range(AG_PER_IMG):
            aug_img = augment(img)
            filename = f"aug_{count:05d}.png"
            aug_img.save(os.path.join(OUTPUT_DIR, filename))
            count += 1

print(f"Selesai! Total gambar augmentasi: {count}")