#
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
model = load_model('fashion_mnist_model.h5')
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
st.title('Fashion MNIST Classifier- by Siri')
st.write('Upload an image to get prediction...')
uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(28, 28), color_mode='grayscale')
    image = img_to_array(image).reshape(1, 28, 28, 1).astype('float32') / 255
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write(f'Prediction: {class_names[predicted_class]}')
