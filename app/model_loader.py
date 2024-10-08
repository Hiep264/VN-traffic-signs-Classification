import torch
from transformers import AutoConfig, AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import io
import json

class ModelLoader:
    def __init__(self, model_path: str, label_names_path: str):
        self.model_path = model_path
        self.label_names_path = label_names_path
        self.model, self.image_processor, self.label_names = self.load_model()

    def load_model(self):
        model = None
        image_processor = None
        label_names = None

        try:
            config = AutoConfig.from_pretrained(self.model_path)
            model = AutoModelForImageClassification.from_pretrained(self.model_path, config=config)
            image_processor = AutoImageProcessor.from_pretrained(self.model_path)
            print("Model and processor loaded successfully.")

            # Load label names
            with open(self.label_names_path, 'r') as f:
                label_names = json.load(f)
            print("Label names loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

        return model, image_processor, label_names

    def predict_image(self, image: Image.Image):
        # Kiểm tra xem model và image_processor có được khởi tạo thành công không
        if self.model is None or self.image_processor is None:
            raise RuntimeError("Failed to load model or image processor.")

        inputs = self.image_processor(images=image, return_tensors="pt")  # Đảm bảo sử dụng đúng image processor

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        predicted_class_idx = outputs.logits.argmax(-1).item()  # Chỉ số lớp được dự đoán
        predicted_label = self.label_names[str(predicted_class_idx)]  # Ánh xạ đến tên lớp

        return predicted_class_idx, predicted_label  # Trả về cả chỉ số lớp và tên lớp
