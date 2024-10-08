from fastapi import FastAPI, File, UploadFile
from model_loader import ModelLoader  # Sử dụng lớp ModelLoader đã được tạo
import uvicorn
from fastapi.responses import HTMLResponse
from io import BytesIO
from PIL import Image

app = FastAPI()

# Khởi tạo ModelLoader
model_path = '/mnt/d/ViT/deploy/swin_model/swin_model'
label_names_path = '/mnt/d/ViT/deploy/label_names.json'
model_loader = ModelLoader(model_path, label_names_path)

@app.get('/', response_class=HTMLResponse)
async def read_root():
    
    with open('index.html', 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    print("Image bytes received, processing...")

    image = Image.open(BytesIO(image_bytes))
    predicted_class_idx, predicted_label = model_loader.predict_image(image)

    # Log kết quả trả về trước khi gửi phản hồi
    prediction_result = {"predicted_class_index": predicted_class_idx, "predicted_class_label": predicted_label}
    print(f"Prediction result: {prediction_result}")

    return prediction_result  # Trả về JSON dưới dạng từ điển


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)
