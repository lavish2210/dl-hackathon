from django.shortcuts import render
from django.http import JsonResponse 

def home(request): 
    return render(request, 'home.html')

def get_image(request):
    return JsonResponse({'image_url':'image.jpg'}, safe=False)

def about(request):
    return render(request,'about.html')