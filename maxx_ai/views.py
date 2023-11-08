
from django.http import JsonResponse
import os
from django.shortcuts import redirect, render
from maxx_ai import settings
from maxx_ai.models import UploadedFile
from maxx_ai.forms import FileUploadForm
from django.shortcuts import render
from django.shortcuts import render
import requests
import json
from nudenet import NudeDetector
from django.http import HttpResponse
# from nudenet import NudeClassifier


def home(request):
    return render(request, "home2.html", {})


def upload_view(request):
    return render(request, "index.html", {})
    




def upload_file(request):
    if request.method == 'POST':
        form = UploadedFile()
        form.desc = request.POST['desc']
        form.file = request.FILES.get('file', 'null')
        if form is not None:
            form.save()
            return redirect('home')
    else:
        form = FileUploadForm()
    current_user = request.user
    context = {"name": current_user }
    return render(request, 'maxx_ai/home2.html', {'form': form}, context)


def get_file(request):
    data = request.FILES.get('file')
    
    nude_detector = NudeDetector()
    nude_detector.detect('C:/Users/Dara/Downloads/Telegram Desktop/IMG_4557.JPG')

def get_file(request):
    
    if request.method == 'POST':
        form = UploadedFile()
        form.desc = request.POST['desc']
        form.file = request.FILES.get('file', 'null')
        if form is not None:
            form.save()
    else:
        form = FileUploadForm()
    
    nude_detector = NudeDetector()
    data = request.FILES.get('file')
    data = str(data)
    path = "D:/School Work/maxx_ai/media/"
    print(path+data)
    detections = nude_detector.detect(path+data)
    print(detections)
    return HttpResponse("File processed successfully!")


