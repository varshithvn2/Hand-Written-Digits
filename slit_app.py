import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the trained model
model = tf.keras.models.load_model('handwritten.model')

# Streamlit app
st.markdown("<h1 style='text-align: center; color: black;'>Hand Written Digit Recognition</h1>", unsafe_allow_html=True)
st.write("Fill the canvas, draw in the center")

# Specify canvas parameters in the application
drawing_mode = "freedraw"  # Set drawing mode to freedraw
stroke_width = 18
stroke_color = "#FFFFFF"  # Set stroke color to white for a black canvas
bg_color = "#000000"  # Set background color to black
realtime_update = True  # Update in real-time

# Create a canvas component
canvas_result = st_canvas(
    fill_color=bg_color,
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=250,
    width=250,
    drawing_mode=drawing_mode,
    key="canvas",
)

# Add a "Recognize" button with styling
if st.button("Recognize", key="recognize_button", help="Click to recognize the digit"):
    # Convert the canvas image to NumPy array
    image_array = np.array(canvas_result.image_data)

    # Preprocess the image for the model
    img_array = np.asarray(Image.fromarray(image_array).convert("L").resize((28, 28))).reshape(1, 28 * 28) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Display the prediction with improved formatting
    st.subheader(f"Prediction Result: **{predicted_class}**")
