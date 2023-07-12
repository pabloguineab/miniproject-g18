import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Image Recognition",
    layout="centered",
    initial_sidebar_state="collapsed"
)

@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

IMAGE_SIZE = 224
CHANNELS = 3

class_indices_flower = {'daisy': 0, 'dandelion': 1, 'rose': 2, 'sunflower': 3, 'tulip': 4}
class_names_flower = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

class_indices_dog = {'beagle': 0, 'chihuahua': 1, 'doberman': 2, 'french_bulldog': 3, 'golden_retriever': 4, 'malamute': 5, 'pug': 6, 'saint_bernard': 7, 'scottish_deerhound': 8, 'tibetan_mastiff': 9}
class_names_dog = list(class_indices_dog.keys())

def homepage(title, link, examples):
    st.header(title)
    st.subheader("Dataset")
    with st.expander("Link"):
        st.write(f"[Kaggle]({link})")

    with st.expander("Examples"):
        for i in range(0, len(examples), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(examples):
                    with cols[j]:
                        st.image(f"datasamples/{examples[i+j]}.jpg", width=200, caption=examples[i+j])

    with st.expander("Splits"):
        st.write("**Overall Samples:** 4318")
        st.write("**Training:** 70%")
        st.write("**Validaiton:** 20%")
        st.write("**Testing:** 10%")
    
    with st.expander("Data Augmentation"):
        code = """
        ImageDataGenerator(
            rescale = 1./255,
            rotation_range = 25,
            shear_range = 0.5,
            zoom_range = 0.5,
            width_shift_range = 0.2,
            height_shift_range=0.2,
            horizontal_flip=True
            )
        
        """
        st.code(code, language='python')
    
    with st.expander("Image Shape"):
        st.write((IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    
    st.subheader("Model Architecture")

    with st.expander("Feature Extraction"):
        code = """
mobile_net = MobileNet(
    weights = 'imagenet', 
    include_top = False, 
    input_shape = IMG_SHAPE)
        
for layer in mobile_net.layers:
            layer.trainable = False
        """
        st.code(code, language='python')

    with st.expander("Final Model"):
        code = """
        model = Sequential([
        mobile_net,
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dense(5, activation='softmax')
        ])
        """
        st.code(code, language='python')

    st.write("**Optimizer:**", "ADAM")
    st.write("**Loss:**", "Categorical Crossentropy")

    st.subheader("Training and Validaiton Loss")
    st.image("loss.png")

    st.subheader("Model Testing")
    code = """
testing = ImageDataGenerator(rescale = 1./255)
test_batches = testing.flow_from_directory(
                    TEST_PATH, 
                    target_size=(IMAGE_SIZE,IMAGE_SIZE), 
                    batch_size=BATCH_SIZE, 
                    class_mode="categorical",
                    seed=42)

results = model.evaluate(test_batches)
    """
    st.code(code, language='python')
    st.write("**Testing Loss:**","0.32296934723854065")
    st.write("**Testing Accuracy:**","0.8812785148620605")

def try_model(model, class_names):
    with st.spinner("Loading Model"):
        uploaded_file = st.file_uploader("Choose an Image", type=['png', 'jpg', 'JPEG'])
        if st.button("Submit"):
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=300)
                with st.spinner("Predicting"):
                    pred = predict(image, model)
                    predicted_class = class_names[np.argmax(pred)]
                st.success(f"I think the image is a **{predicted_class}**")

def predict(image, model):
    data = np.ndarray(shape=(1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.float32)
    size = (IMAGE_SIZE, IMAGE_SIZE)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 255.0)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return prediction

def launch():
    selected = option_menu(
    menu_title=None,
    options = ["Flower Recognition Code", "Try Flower Recognition Model", "Dog Breeds Recognition Code", "Try Dog Breeds Recognition Model", "About"],
    icons= ["code-slash","play", "code-slash","play", "info"],
    menu_icon="list",
    default_index=0,
    orientation="horizontal"
    )

    if selected == "Flower Recognition Code":
        homepage("Flowers Recognition", "https://www.kaggle.com/datasets/alxmamaev/flowers-recognition", list(class_indices_flower.keys()))
    if selected == "Try Flower Recognition Model":
        try_model(load_model('flower-recog-cnn.h5'), class_names_flower)
    if selected == "Dog Breeds Recognition Code":
        homepage("Dog Breeds Recognition", "https://www.kaggle.com/jessicali9530/stanford-dogs-dataset", list(class_indices_dog.keys()))
    if selected == "Try Dog Breeds Recognition Model":
        try_model(load_model('dog-breeds-recog-cnn.h5'), class_names_dog)
    if selected == "About":
        about()

def about():
    st.write("### ADVANCED MATHS DL")
    st.write("GROUP 18")
    st.write("Pablo Guinea Benito",  
             "Joy",
             "Abdullah",
             "Dushyant",
            )
    st.image("George_Brown_College_logo.svg.png", width=350)

if __name__ == "__main__":
    launch()
