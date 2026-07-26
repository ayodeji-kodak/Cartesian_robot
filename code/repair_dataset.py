import os
import shutil
from pathlib import Path

# =====================================
# CHANGE THIS TO YOUR DATASET LOCATION
# =====================================

DATASET_PATH = r"/Users/ayodeji/Downloads/rail-cars"

# Example:
# DATASET_PATH = r"/Users/ayodeji/Downloads/rail-cars"

# =====================================

folders = [
    "train/labels",
    "valid/labels",
    "test/labels"
]

print("="*60)
print("Creating backup...")
print("="*60)

backup = DATASET_PATH + "_backup"

if not os.path.exists(backup):
    shutil.copytree(DATASET_PATH, backup)
    print(f"Backup created:\n{backup}")
else:
    print("Backup already exists.")

print()

total_files = 0
total_labels = 0

for folder in folders:

    label_dir = os.path.join(DATASET_PATH, folder)

    if not os.path.exists(label_dir):
        print(f"Skipping {folder}")
        continue

    print(f"Repairing {folder}")

    for txt in Path(label_dir).glob("*.txt"):

        total_files += 1

        new_lines = []

        with open(txt, "r") as f:
            lines = f.readlines()

        for line in lines:

            line = line.strip()

            if line == "":
                continue

            parts = line.split()

            # Replace ANY class with class 0
            parts[0] = "0"

            new_lines.append(" ".join(parts))

            total_labels += 1

        with open(txt, "w") as f:
            f.write("\n".join(new_lines))

print()
print("="*60)
print("Repair complete!")
print("="*60)

print(f"Files modified : {total_files}")
print(f"Labels changed : {total_labels}")

# =====================================
# Create a clean data.yaml
# =====================================

yaml = f"""train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1

names:
  0: tank_wagon
"""

with open(os.path.join(DATASET_PATH, "data.yaml"), "w") as f:
    f.write(yaml)

print()
print("New data.yaml created.")
print("Dataset is ready for training.")
