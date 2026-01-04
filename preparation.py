import os
import shutil
import random

dataset_dir = 'C:/ML/Projek/cnn_alfabet/dataset/augmented_images1'
output_dir = 'C:/ML/Projek/cnn_alfabet/dataset/dataset_alphabet'

for split in ['train', 'valid', 'test']:
    split_path = os.path.join(output_dir, split)
    os.makedirs(split_path, exist_ok=True)

for label in os.listdir(dataset_dir):
    label_path = os.path.join(dataset_dir, label)
    if not os.path.isdir(label_path):
        continue
    
    images = os.listdir(label_path)
    random.shuffle(images)  # shuffle data
    
    n_total = len(images)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val
    
    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train+n_val],
        'test': images[n_train+n_val:]
    }
    
    for split_name, split_files in splits.items():
        split_label_dir = os.path.join(output_dir, split_name, label)
        os.makedirs(split_label_dir, exist_ok=True)
        
        for f in split_files:
            src = os.path.join(label_path, f)
            dst = os.path.join(split_label_dir, f)
            shutil.copy2(src, dst)

print("Dataset berhasil dibagi menjadi train, val, test!")

# import os
# import shutil as sh
# import random as rd
# from sklearn.model_selection import train_test_split

# path_dataset = "dataset/dataset_asli"
# path_dataset_fix = "dataset/dataset_fix"
# max_file_perkelas = 3000

# for split in ("train", "valid", "test"):
#     os.makedirs(os.path.join(path_dataset_fix, split), exist_ok=True)

# classes = [d for d in os.listdir(path_dataset) if os.path.isdir(os.path.join(path_dataset, d))]

# for cls in classes:
#     src_path = os.path.join(path_dataset, cls)
#     all_files = os.listdir(src_path)
    
#     # Ambil Sampel Acak
#     if len(all_files) > max_file_perkelas:
#         all_files = rd.sample(all_files, max_file_perkelas)
    
#     # Split Data
#     train_files, temp_files = train_test_split(
#         all_files, test_size=0.2, random_state=42
#     )
    
#     val_files, test_files = train_test_split(
#         temp_files, test_size=0.5, random_state=42
#     )

#     # Pindahin File ke Folder
#     def move_files(files, split_name):
#         dest_path = os.path.join(path_dataset_fix, split_name, cls)
#         os.makedirs(dest_path, exist_ok=True)
#         for f in files:
#             sh.copy(os.path.join(src_path, f), os.path.join(dest_path, f))

#     move_files(train_files, 'train')
#     move_files(val_files, 'valid')
#     move_files(test_files, 'test')
    
#     print(f"Selesai memproses Kelas {cls}: {len(all_files)} gambar.")

# print(f"\nSukses! Dataset tersimpan di: {path_dataset_fix}")