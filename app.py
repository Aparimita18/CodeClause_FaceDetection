from mtcnn.mtcnn import MTCNN
import streamlit as st
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from background import set_background_image
set_background_image("file:///D:/college%20project/face%20bg.jpg.webp")


st.header("Face Detection using a Pre-trained CNN model")

choice = st.selectbox("",[
    "Face Detection - Show Bounding Box",
    "Face Detection - Extract Face",
    "Face Verification"
])
def main():
    fig = plt.figure()
    if choice == "Face Detection - Show Bounding Box":
        st.subheader("Face Detection - Show Bounding Box")
        st.write("Please upload an image containing a face. A box will be drawn highlighting the face using the pretrained VGGFace model.")
        # load the image
        uploaded_file = st.file_uploader("Upload Image", type=["jpg","png"], key="1")
        if uploaded_file is not None:
            data = asarray(Image.open(uploaded_file))
            # plot the image
            plt.axis("off")
            plt.imshow(data)
            # get the context for drawing boxes
            ax = plt.gca()
            # plot each box
            # load image from file
            # create the detector, using default weights
            detector = MTCNN()
            # detect faces in the image
            faces = detector.detect_faces(data)
            for face in faces:
                # get coordinates
                x, y, width, height = face['box']
                # create the shape
                rect = Rectangle((x, y), width, height, fill=False, color='green')
                # draw the box
                ax.add_patch(rect)
                # draw the dots
                for _, value in face['keypoints'].items():
                    # create and draw dot
                    dot = Circle(value, radius=2, color='green')
                    ax.add_patch(dot)
            # show the plot
            st.pyplot(fig)
        st.write("The box highlights the face and the dots highlight the identified features.")

    elif choice == "Face Detection - Extract Face":
        st.subheader("Face Detection - Extract Face")
        st.write("Please upload an image containing a face. The part of the image containing the face will be extracted using the pretrained VGGFace model.")

        uploaded_file = st.file_uploader("Upload Image", type=["jpg","png"], key="2")
        if uploaded_file is not None:
            column1, column2 = st.columns(2)
            image = Image.open(uploaded_file)
            with column1:
                size = 450, 450
                resized_image = image.thumbnail(size)
                image.save("thumb.png")
                st.image("thumb.png")
            pixels = asarray(image)
            plt.axis("off")
            plt.imshow(pixels)
            # create the detector, using default weights
            detector = MTCNN()
            # detect faces in the image
            results = detector.detect_faces(pixels)
            # extract the bounding box from the first face
            x1, y1, width, height = results[0]["box"]
            x2, y2 = x1 + width, y1 + height
            # extract the face
            face = pixels[y1:y2, x1:x2]
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize((224, 224)) # Rodgers -> You can just save this as image
            face_array = asarray(image)
            with column2:
                 plt.imshow(face_array)
                 st.pyplot(fig)
        
    elif choice == "Face Verification":
        st.subheader("Face Verification")
        st.write("Please upload two image of the same person. The model will check if the two images contain the same face.")
        st.write("Classifies whether the images match based on the probability score predicted by the model. If the difference is below the threshold (0.5 here) then the images are said to be identical.")
        column1, column2 = st.columns(2)
    
        with column1:
            image1 = st.file_uploader("Upload First Image", type=["jpg","png"], key="3")
           
        with column2:
            image2 = st.file_uploader("Upload Second Image", type=["jpg","png"], key="4")
        # define filenames
        if (image1 is not None) & (image2  is not None):
            col1, col2 = st.columns(2)
            image1 =  Image.open(image1)
            image2 =  Image.open(image2)
            with col1:
                st.image(image1)
            with col2:
                st.image(image2)

            filenames = [image1,image2]

            faces = [extract_face(f) for f in filenames]
            # convert into an array of samples
            samples = asarray(faces, "float32")
            # prepare the face for the model, e.g. center pixels
            samples = preprocess_input(samples, version=2)
            # create a vggface model
            model = VGGFace(model= "resnet50" , include_top=False, input_shape=(224, 224, 3),
            pooling= "avg" )
            # perform prediction
            embeddings = model.predict(samples)
            thresh = 0.5

            score = cosine(embeddings[0], embeddings[1])
            if score <= thresh:
                st.success( " > Face is a match ( Score %.3f <= %.3f) " % (score, thresh))
            else:
                st.error(" > Face is NOT a match ( Score %.3f > %.3f)" % (score, thresh))


def extract_face(file):
    pixels = asarray(file)
    plt.axis("off")
    plt.imshow(pixels)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]["box"]
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = asarray(image)
    return face_array
        


if __name__ == "__main__":
    main()

