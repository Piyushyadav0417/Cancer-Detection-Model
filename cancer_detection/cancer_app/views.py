import os
import uuid
from django.shortcuts import render, redirect
from django.conf import settings
from .forms import PredictionForm
from .ml.xai_predict import generate_visual_prediction


def predict_view(request):
    print('predict_view is calling')
    output_image_url = None

    # 🧼 Clean up previous image if exists
    old_output_file = request.session.get('output_file')
    if old_output_file:
        old_output_path = os.path.join(settings.MEDIA_ROOT, old_output_file)
        if os.path.exists(old_output_path):
            os.remove(old_output_path)
        request.session.pop('output_file')

    if request.method == 'POST':
        form = PredictionForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            model_name = form.cleaned_data['model_choice']

            # Save uploaded image
            input_filename = f"{uuid.uuid4().hex}_{image.name}"
            input_path = os.path.join(settings.MEDIA_ROOT, input_filename)

            with open(input_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            # Output file path
            output_filename = f"output_{uuid.uuid4().hex}.png"
            output_path = os.path.join(settings.MEDIA_ROOT, output_filename)

            # Run prediction + Grad-CAM
            model_path = os.path.join(settings.BASE_DIR, 'cancer_app', 'models', model_name)

            generate_visual_prediction(model_path, input_path, output_path)


            os.remove(input_path)
            
            # Show output image
            output_image_url = settings.MEDIA_URL + output_filename
            request.session['output_file'] = output_filename
            
            return redirect('result_page')

    else:
        form = PredictionForm()

    return render(request, 'cancer_app/predict.html', {
        'form': form
    })
    

def result(request):
    output_filename = request.session.get('output_file')
    output_image_url = None

    if output_filename:
        output_path = os.path.join(settings.MEDIA_ROOT, output_filename)
        if os.path.exists(output_path):
            output_image_url = settings.MEDIA_URL + output_filename

    return render(request, 'cancer_app/result.html', {
        'output_image_url': output_image_url
    })


from .delete import clear_media_folder

def clear_and_redirect_home(request):
    clear_media_folder()
    return redirect('home_page')  # assuming this is your home

def home(request):
    return render(request, 'cancer_app/home.html')

def model_1(request):
    return render(request, 'cancer_app/model_1.html')
def model_2(request):
    return render(request, 'cancer_app/model_2.html')
def input(request):
    print('Innput Calling')
    return render(request, 'cancer_app/input.html')

    
