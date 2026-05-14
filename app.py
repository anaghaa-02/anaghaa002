import streamlit as st
import numpy as np
from PIL import Image, ImageOps

# Try importing tflite runtime based on platform availability
try:
import tensorflow.lite as tflite

# 1. Load your pre-trained TFLite model safely
@st.cache_resource
def load_tflite_model():
    # Make sure 'plant_disease_model.tflite' is uploaded to your GitHub repository folder
    interpreter = tflite.Interpreter(model_path='plant_disease_model.tflite')
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Error loading model. Ensure 'plant_disease_model.tflite' is in your GitHub repo. Details: {e}")

# 2. Define your target classes (Replace these with your exact dataset outputs)
CLASS_NAMES = ['Healthy Leaf', 'Early Blight', 'Late Blight', 'Leaf Mold'] 

# 3. Build the Mobile-Friendly UI Layout
st.set_page_config(page_title="Plant Doctor AI", layout="centered")
st.title("🌱 Plant Disease Detector")
st.write("Take a photo or upload an image of a crop leaf to scan for diseases.")

# File uploader acts as both file selector and camera trigger on mobile
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("🔄 Analyzing leaf...")
    
    # 4. Pre-process image to match your model training configuration (224x224x3)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCEZOS)
    
    # Convert image to RGB if it is in grayscale or RGBA mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    image_array = np.asarray(image)
    
    # Normalize the image pixels (0-1) to match MobileNetV2 training
    normalized_image_array = (image_array.astype(np.float32) / 255.0)
    
    # Reshape to match model input batch shape (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # 5. Run prediction using the TFLite Interpreter
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Calculate highest confidence metrics
    best_match_idx = np.argmax(prediction)
    confidence = prediction[best_match_idx] * 100
    detected_class = CLASS_NAMES[best_match_idx]
    
    # 6. Display results
    st.success(f"**Diagnosis:** {detected_class}")
    st.info(f"**Confidence Level:** {confidence:.2f}%")
