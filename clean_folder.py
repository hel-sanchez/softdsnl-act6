from PIL import Image, UnidentifiedImageError
import os

def clean_folder(path):
    removed = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Will raise exception if not a valid image
            except (UnidentifiedImageError, IOError, OSError) as e:
                print(f"❌ Removing: {file_path} ({e})")
                try:
                    os.remove(file_path)
                    removed += 1
                except Exception as delete_error:
                    print(f"⚠️ Could not delete {file_path}: {delete_error}")
    print(f"\n✅ Done. Removed {removed} corrupted or invalid image(s) from {path}")

# Run cleaner on your train and test folders
clean_folder("data/training_set")
clean_folder("data/testing_set")
