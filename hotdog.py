import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Resizing
from tensorflow.io import decode_image
from tensorflow.keras.utils import img_to_array, load_img

model = load_model('./')
img_size = 128
image_file = None

resize = Sequential([Resizing(img_size, img_size)])

st.title("Team 4: Hot Dog or Not-Hot-Dog?")

# https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader
# type restriction taken from tf.io.decode_image docs that references 
# these file types
upload = st.file_uploader('Upload an image :open_file_folder: :', 
                           accept_multiple_files=False,
                           type=['png', 'jpg', 'jpeg', 'bmp', 'gif']
                          )

# https://docs.streamlit.io/library/api-reference/widgets/st.camera_input
# could get fancy and also do a webcam image
cam = st.camera_input('Use your camera :camera: :')

if st.button('Check for Hot Dog'):
    if upload is None and cam is None:
        st.write("Please upload an image :open_file_folder: or take a picture :camera:")
    elif upload is not None:
        # execute uploaded image code here
        image_file = upload

         # Convert single image to a batch.
        # input_arr = img_to_array(image).reshape((None, img_size, img_size, 3))
        # predictions = model.predict(input_arr)
        
    elif cam is not None:
        # execute cam code here
        #st.write("You have taken a picture")

        image_file = cam
        # bytes_data = cam.getvalue()
        # img_tensor = tf.io.decode_image(bytes_data, channels=3)
        # img = Image.open(cam)
        # img_array = ds(img)
        # img_array = resize(img_array)

    if image_file:
        st.image(image_file)
        image = load_img(image_file, target_size=(img_size, img_size, 3))
        input_arr = img_to_array(image)
        input_arr = np.array([input_arr]) 
        pred = np.where(model.predict(input_arr) < 0.5, "Hot Dog", "Not Hot Dog")
        # model.predict(input_arr)
        st.write(f'Our model believes this image is {pred[0][0]}')