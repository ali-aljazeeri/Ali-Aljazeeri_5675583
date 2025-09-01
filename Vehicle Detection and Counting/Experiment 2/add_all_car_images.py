import os
import shutil
from PIL import Image

# all source directories
source_dirs = [
    './Cars object detection /DATA/DATA/test',
    './Cars object detection /DATA/DATA/train',
    './Cars Detection/valid/images',
    './Cars Detection/train/images',
    './Cars Detection/test/images',
    './01-whole'
]

# output
output_dir = 'combined_car_images/images'
os.makedirs(output_dir, exist_ok=True)

# image types
image_extensions = ('.jpg', '.jpeg', '.png')

# track of unique prefix groups, as noticed some images are duplicated but have same prefix
seen_prefixes = set()
counter = 1

# Loop through all source folders
for source_dir in source_dirs:
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
            prefix = filename  # fallback if no .rf.

        if prefix in seen_prefixes:
            continue  # Skip duplicate

        try:
            # image validation
            with Image.open(file_path) as img:
                img.verify()

            # copy image to output with ordered name
            new_filename = f"image_{counter:03d}{os.path.splitext(filename)[1]}"
            new_path = os.path.join(output_dir, new_filename)

            shutil.copy(file_path, new_path)
            print(f"Copied {filename} as {new_filename}")

            seen_prefixes.add(prefix)
            counter += 1

        except Exception as e:
            print(f"Skipping {filename}: invalid image ({e})")

print(f"\nTotal unique images copied: {counter - 1}")
