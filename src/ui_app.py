import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
from model_loader import ModelHandler

class ImageClassifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("CIFAR-10 Image Classifier")
        self.geometry("700x600")
        self.model_handler = ModelHandler()
        self.image_path = None

        self.title_label = ctk.CTkLabel(self, text="Deep Learning Image Classifier", font=("Arial", 24, "bold"))
        self.title_label.pack(pady=20)

        self.image_frame = ctk.CTkFrame(self, width=400, height=300)
        self.image_frame.pack(pady=10)
        self.image_label = ctk.CTkLabel(self.image_frame, text="No image uploaded", width=400, height=300)
        self.image_label.pack()

        self.upload_button = ctk.CTkButton(self, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=15)

        self.predict_button = ctk.CTkButton(self, text="Predict", command=self.predict)
        self.predict_button.pack(pady=10)

        self.result_label = ctk.CTkLabel(self, text="", font=("Arial", 18))
        self.result_label.pack(pady=10)
        self.prob_label = ctk.CTkLabel(self, text="", font=("Arial", 16))
        self.prob_label.pack(pady=5)

    def upload_image(self):
        """Open file dialog to upload image"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if file_path:
            self.image_path = file_path
            image = Image.open(file_path).resize((300, 300))
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
            self.result_label.configure(text="")
            self.prob_label.configure(text="")

    def predict(self):
        """Predict uploaded image"""
        if not self.image_path:
            self.result_label.configure(text="Please upload an image first.")
            return
        label, confidence = self.model_handler.predict(self.image_path)
        self.result_label.configure(text=f"Prediction: {label}")
        self.prob_label.configure(text=f"Confidence: {confidence*100:.2f}%")

if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    app = ImageClassifierApp()
    app.mainloop()
