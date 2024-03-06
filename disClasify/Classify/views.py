import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.shortcuts import render
from .forms import ImageUploadForm

img_size = (224, 224)

class_names = ["Apple Scab", "Blue Berry (healthy)", "Cherry Powdery Mildew", "Corn Common rust",
               "Corn Nortern Leaf Blight", "Grape Black Rot", "Grape Healthy", "Orange Citrus Greening",
               "Peach Bacterial spot", "Peach Healthy", "Pepper Bell Bacterial spot", "Pepper Bell Healthy",
               "Potato Late Blight", "Raspberry Healthy", "Soybean Healthy", "Squash Powdery Mildew",
               "Strawberry Healthy", "Strawberry Leaf Scorch", "Tomato Leaf Mold", "Tomato Mosaic virus",]

rm = [
    "For Apple Scab, apply a fungicide-only spray like myclobutanil (Spectracide Immunox Multipurpose Fungicide Spray Concentrate). Start spraying from the green tip until after petal fall. Maintain good orchard hygiene.",
    "No specific remedy needed for a healthy Blueberry plant. Ensure proper watering and nutrient balance.",
    "To treat Cherry Powdery Mildew, use fungicides containing sulfur or potassium bicarbonate. Improve air circulation around the cherry trees by proper pruning.",
    "For Corn Common Rust, treat with fungicides containing azoxystrobin or propiconazole. Remove and destroy infected plants to prevent further spread.",
    "To control Corn Northern Leaf Blight, use fungicides like chlorothalonil or azoxystrobin. Implement crop rotation and sanitation practices to reduce disease pressure.",
    "For Grape Black Rot, apply fungicides like myclobutanil or captan. Remove and destroy infected fruit and leaves. Prune grapevines for better air circulation.",
    "Maintain overall grapevine health by proper pruning and applying fungicides if necessary. Ensure good air circulation in the vineyard.",
    "For Orange Citrus Greening, manage the disease with insecticides to control the Asian citrus psyllid vector. Apply nutritional sprays to support tree health.",
    "Treat Peach Bacterial Spot with copper-based sprays during the dormant season. Prune peach trees to enhance air circulation and reduce disease pressure.",
    "No specific remedy needed for a healthy Peach tree. Implement proper pruning and maintenance practices.",
    "To control Pepper Bell Bacterial Spot, use copper-based sprays. Apply preventive treatments and practice crop rotation to minimize disease recurrence.",
    "No specific remedy needed for a healthy Pepper Bell plant. Focus on proper watering and nutrition.",
    "For Potato Late Blight, use fungicides containing chlorothalonil or copper. Harvest potatoes early and store them in cool, dry conditions to prevent disease development.",
    "No specific remedy needed for a healthy Raspberry plant. Ensure proper irrigation and soil fertility.",
    "No specific remedy needed for a healthy Soybean plant. Implement crop rotation to manage soil-borne diseases.",
    "For Squash Powdery Mildew, use fungicides like potassium bicarbonate. Alternatively, apply organic solutions such as neem oil or sulfur for control.",
    "No specific remedy needed for a healthy Strawberry plant. Maintain good strawberry bed hygiene and proper irrigation.",
    "To treat Strawberry Leaf Scorch, remove and destroy infected leaves. Apply copper-based fungicides preventively. Enhance air circulation in the strawberry patch.",
    "For Tomato Leaf Mold, use fungicides containing chlorothalonil or copper. Ensure proper spacing between tomato plants for improved air circulation.",
    "For Tomato Mosaic Virus, control aphids and use resistant tomato varieties. Remove and destroy infected plants promptly to prevent further spread."
]

model = load_model('D:\Josika\project\disease-classification\plant_disease_model.h5')

def load_and_predict(image_path):
    global model
    if not os.path.isfile(image_path):
        return JsonResponse({"error": "File not found."}, status=400)

    img = image.load_img(image_path, target_size=img_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    return predicted_class

def save_and_predict(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image to the media folder
            image_file = form.cleaned_data['image']
            file_name = default_storage.save(os.path.join(settings.MEDIA_ROOT, image_file.name), ContentFile(image_file.read()))

            # Get the full path of the saved image
            image_path = os.path.join(settings.MEDIA_ROOT, file_name)

            # Predict the class of the image
            predicted_class = load_and_predict(image_path)
            cont ={
                'result':class_names[predicted_class],
                'rem':rm[predicted_class]
            }

            return render(request, 'result.html', cont)
    else:
        form = ImageUploadForm()

    return render(request, 'upload_image.html', {'form': form})