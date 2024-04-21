import streamlit as st
import tensorflow as tf
from PIL import Image

# Load the Keras model
model = tf.keras.models.load_model('./cats_vs_dogs_model.h5')

# Function to preprocess the image

def preprocess_image(image):
    # Ensure image has 3 channels (RGB)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Resize image to match model input size
    img = image.resize((150, 150))
    # Convert image to array and normalize pixel values
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    return img_array


st.title('Cat vs Dog Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)

    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(image)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Make predictions
    try:
        prediction = model.predict(img_array)

        # Calculate confidence levels in percentage
        dog_confidence = prediction[0][0] * 100
        cat_confidence = (1 - prediction[0][0]) * 100

        st.write(f'Confidence - Dog: {dog_confidence:.2f}%')
        st.write(f'Confidence - Cat: {cat_confidence:.2f}%')

        # Interpret prediction
        predicted_class = 'Cat' if prediction[0][0] < 0.5 else 'Dog'

        st.write(f'Prediction: {predicted_class}')

    except Exception as e:
        st.error(f"Error encountered: {e}")
