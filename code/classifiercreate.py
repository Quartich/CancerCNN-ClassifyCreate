import streamlit as st
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

#title
st.title("CNN Image Classification and Training Tool")

#sidebar
st.sidebar.header("Configuration")
app_mode = st.sidebar.radio("Choose mode", ["Train a Model", "Load and Classify"])

if app_mode == "Train a Model":
    #all the directory stuff
    input_dir = st.text_input("Dataset Directory", "./dataset")
    output_dir = st.text_input("Preprocessed Directory", "./preprocessed_data")
    classes = st.text_input("Class Names (comma-separated)", "class0,class1,class2").split(',')
    crop_size = st.slider("Crop Size", 128, 512, 384)
    max_images = st.number_input("Max Images per Class", min_value=10, max_value=10000, value=1000)
    epochs = st.slider("Epochs", 1, 50, 20)
    input_shape = st.slider("Input Shape", 128, 512, 384)
    neurons = st.slider("Neurons in Dense Layer", 32, 512, 128)
    model_name = st.text_input("Model Save Name", "trained_model.h5")

    if st.button("Start Preprocessing"): #preprocessing
        def preprocess_and_limit_images(input_dir, output_dir, classes, crop_size, max_images):
            for class_name in classes:
                input_class_dir = os.path.join(input_dir, class_name)
                output_class_dir = os.path.join(output_dir, class_name)
                os.makedirs(output_class_dir, exist_ok=True)

                all_images = [f for f in os.listdir(input_class_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
                selected_images = random.sample(all_images, min(max_images, len(all_images)))

                for image_name in selected_images:
                    input_path = os.path.join(input_class_dir, image_name)
                    output_path = os.path.join(output_class_dir, image_name)

                    try:
                        with Image.open(input_path) as img:
                            if img.size != (crop_size, crop_size):
                                continue
                            left = random.randint(0, img.width - crop_size)
                            top = random.randint(0, img.height - crop_size)
                            right = left + crop_size
                            bottom = top + crop_size
                            img_cropped = img.crop((left, top, right, bottom))
                            img_cropped.save(output_path)
                    except Exception as e:
                        st.error(f"Error processing {input_path}: {e}")
        preprocess_and_limit_images(input_dir, output_dir, classes, (crop_size, crop_size), max_images)
        st.success("Preprocessing completed!")

    if st.button("Start Training"):  #training
        datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.1)

        train_gen = datagen.flow_from_directory(
            output_dir, target_size=(input_shape, input_shape), batch_size=32, class_mode='categorical', subset='training')
        val_gen = datagen.flow_from_directory(
            output_dir, target_size=(input_shape, input_shape), batch_size=32, class_mode='categorical', subset='validation')

        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(input_shape, input_shape, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(neurons, activation='relu'),
            Dropout(0.5),
            Dense(len(classes), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[early_stopping])

        st.success("Training completed!")
        model.save(model_name)
        st.success(f"Model saved as {model_name}")

        #graphs/visualiztion
        st.subheader("Training History")
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].plot(history.history['accuracy'], label='Train Accuracy')
        ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax[0].legend()
        ax[0].set_title("Accuracy")
        ax[1].plot(history.history['loss'], label='Train Loss')
        ax[1].plot(history.history['val_loss'], label='Validation Loss')
        ax[1].legend()
        ax[1].set_title("Loss")
        st.pyplot(fig)

elif app_mode == "Load and Classify":
    #any file ending in .h5
    model_dir = '.'
    models = [f for f in os.listdir(model_dir) if f.endswith('.h5')]

    selected_model = st.selectbox("Select a Model", models)
    if selected_model:
        model_path = os.path.join(model_dir, selected_model)
        model = load_model(model_path)
        st.write(f"Loaded model: **{selected_model}**")
        input_shape = model.input_shape[1:3]  #this gets resolution
        num_classes = model.output_shape[-1]  #can get number of classes
        
        #auto class names.  meant to make a user label option
        classes = [f'Class {i}' for i in range(num_classes)]

        st.write(f"Model expects input shape: {input_shape}")
        st.write(f"Number of classes: {num_classes}")
        uploaded_file = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            img = load_img(uploaded_file, target_size=input_shape)  #preprocessing and resizing for input shape
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array, verbose=0)[0]
            normalized_predictions = predictions / np.sum(predictions)  #makes a percent instead of arbitray number
            adjusted_predictions = normalized_predictions.copy()
            max_idx = np.argmax(adjusted_predictions)

            if adjusted_predictions[max_idx] > 0.9999:  #lol, this makes it so nothing is %100 percent confident (even if the model is).  For presentation purposes
                adjusted_predictions[max_idx] = 0.9999
                remaining_sum = 0.0001
                other_classes = [i for i in range(len(adjusted_predictions)) if i != max_idx]
                for i in other_classes:
                    adjusted_predictions[i] = remaining_sum / len(other_classes)

            prediction_percentages = adjusted_predictions * 100  #predict and display
            predicted_class_idx = np.argmax(adjusted_predictions)
            predicted_class = classes[predicted_class_idx]

            st.write(f"### Predicted Class: **{predicted_class}**")
            st.write("#### Class Confidence Percentages:")
            for cls, prob in zip(classes, prediction_percentages):
                st.write(f"- **{cls}**: {prob:.2f}%")
