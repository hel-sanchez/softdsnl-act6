import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Get the absolute path to the model file, assuming it's in the 'classifier' app directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # classifier app directory
MODEL_PATH = os.path.join(BASE_DIR, 'kaggle_cnn_model.h5')

# Load model only once when the server starts
model = load_model(MODEL_PATH)

class_names = ['Cat', 'Dog']  # adjust if needed

class PredictAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=400)

        try:
            image_file = request.FILES['image']
            image = Image.open(image_file)
            image = image.convert('RGB')
            image = image.resize((64, 64))  # match your model input size

            image_array = img_to_array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            prediction = model.predict(image_array)
            predicted_class = class_names[np.argmax(prediction)]

            return Response({'prediction': predicted_class})

        except Exception as e:
            return Response({'error': str(e)}, status=500)
