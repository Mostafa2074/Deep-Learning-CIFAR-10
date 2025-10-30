import os
import subprocess
import sys
import time

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "my_model.keras")
    src_dir = os.path.join(base_dir, "src")

    if not os.path.exists(model_path):
        print("Model not found. Training a new one...")
        train_script = os.path.join(src_dir, "train_model.py")
        try:
            subprocess.run([sys.executable, train_script], check=True)
            print("Model trained successfully.")
        except subprocess.CalledProcessError:
            print("Error occurred during training.")
            return

    else:
        print("Model already exists — skipping training.")

    print("Launching GUI...")
    ui_script = os.path.join(src_dir, "ui_app.py")

    try:
        subprocess.run([sys.executable, ui_script], check=True)
    except subprocess.CalledProcessError:
        print("Error launching the GUI.")
        return

if __name__ == "__main__":
    main()
