import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

st.title("Plant Disease Detector")
uploaded_file = st.camera_input("Take a photo of the plant leaf")

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    try:
        interpreter = tflite.Interpreter(model_path="plant_model.tflite")
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details['index'])
        
        labels = ['Healthy', 'Blight', 'Rust']
        st.success(f"Prediction: {labels[np.argmax(prediction)]}")
    except Exception as e:
        st.error(f"Model error: {e}")
