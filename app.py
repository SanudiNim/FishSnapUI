from fastapi import FastAPI
import requests
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import uuid
import os

save_directory = "path/to/save/images"
app = FastAPI()

# Load your pre-trained model
model = load_model("model/my_model.h5")

def download_image(url, save_directory):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Generate a random filename with a random salt
        random_salt = str(uuid.uuid4())[:8]
        file_name = f"image_{random_salt}.jpg"
        save_path = os.path.join(save_directory, file_name)
        
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f"Image downloaded successfully and saved at: {save_path}")
        return save_path
    
    except Exception as e:
        print(f"Error downloading image: {e}")


@app.get("/predict/")
async def predict_image(image_url: str):
    image_path = download_image(image_url,"images/")

    img = image.load_img(image_path, target_size=(224, 224))  # Resize image to match MobileNet input size
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # Preprocess the image
    predictions = model.predict(x)

    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Replace these labels with your actual class labels
    class_labels = [
        "Bangus",
        "Big Head Carp",
        "Black Spotted Barb",
        "CatFish",
        "Climbing Perch",
        "FourFinger Thredfin",
        "FreshWater Eel",
        "Glass Perchlet",
        "Goby",
        "Gold Fish",
        "Gourami",
        "Grass Crap",
        "Green Spotted Puffer",
        "Indian Crap",
        "Indo-Pacific Tarpon",
        "Jaguar Gapote",
        "Janitor Fish",
        "Knife Fish",
        "Long- Snouted PipeFish",
        "Mosquito Fish",
        "MudFish",
        "Mullet",
        "Pangasius",
        "Perch",
        "Scat Fish",
        "Silver Barb",  
        "Silver Carp",
        "Silver Perch",
        "Snakehead",
        "TenPounder",
        "Thilapia",]
    
    predicted_label = class_labels[predicted_class_index]

    return {"index": predicted_class_index, "predicted_label": predicted_label}