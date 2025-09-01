import os
import shutil
from PIL import Image

# all source directories
source_dirs = [
    './Cars Detection/valid/images',
    './Cars Detection/train/images',
    './Cars Detection/test/images',
]

# label directories mapped to source dirs
label_dirs = [
    './Cars Detection/valid/labels',
    './Cars Detection/train/labels',
    './Cars Detection/test/labels',
]

# output
output_dir_img = 'car_images/images'
output_dir_lab = 'car_images/labels'
os.makedirs(output_dir_img, exist_ok=True)
os.makedirs(output_dir_lab, exist_ok=True)

# image types
image_extensions = ('.jpg', '.jpeg', '.png')

# track of unique prefix groups, as noticed some images are duplicated but have same prefix
seen_prefixes = set()
counter = 1

# Loop through all source folders
for source_dir, label_dir in zip(source_dirs, label_dirs):
    for filename in sorted(os.listdir(source_dir)):
        if not filename.lower().endswith(image_extensions):
            continue

        file_path = os.path.join(source_dir, filename)

        if not os.path.isfile(file_path):
            continue

        # Extract prefix before .rf.
        if ".rf." in filename:
            prefix = filename.split(".rf.")[0]
        else:
            prefix = os.path.splitext(filename)[0]  # fallback

        if prefix in seen_prefixes:
            continue  # Skip duplicate

        try:
            # image validation
            with Image.open(file_path) as img:
                img.verify()

            # copy image to output with ordered name
            new_filename = f"image_{counter:03d}{os.path.splitext(filename)[1]}"
            new_img_path = os.path.join(output_dir_img, new_filename)
            shutil.copy(file_path, new_img_path)

            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(label_dir, label_filename)
            new_label_filename = f"image_{counter:03d}.txt"
            new_label_path = os.path.join(output_dir_lab, new_label_filename)
            shutil.copy(label_path, new_label_path)

            print(f"Copied {filename} as {new_filename}")

            seen_prefixes.add(prefix)
            counter += 1

        except Exception as e:
            print(f"Skipping {filename}: invalid image ({e})")

print(f"\nTotal unique images copied: {counter - 1}")

