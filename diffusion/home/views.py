from django.shortcuts import render
from django.http import JsonResponse 
from .GenModel import DDPM
from os.path import exists
from torch import tensor, no_grad, load
import matplotlib.pyplot as plt
import os

alphanumericData = {'#': 0,
    '$': 1,
    '&': 2,
    '0': 3,
    '1': 4,
    '2': 5,
    '3': 6,
    '4': 7,
    '5': 8,
    '6': 9,
    '7': 10,
    '8': 11,
    '9': 12,
    '@': 13,
    'A': 14,
    'B': 15,
    'C': 16,
    'D': 17,
    'E': 18,
    'F': 19,
    'G': 20,
    'H': 21,
    'I': 22,
    'J': 23,
    'K': 24,
    'L': 25,
    'M': 26,
    'N': 27,
    'P': 28,
    'Q': 29,
    'R': 30,
    'S': 31,
    'T': 32,
    'U': 33,
    'V': 34,
    'W': 35,
    'X': 36,
    'Y': 37,
    'Z': 38
}

def home(request): 
    return render(request, 'home.html')

def get_image(request):
    var = request.GET.get('input_text')
    
    if (var >= 'a' or var <= 'z'):
        var = var.upper()
    
    num_classes = 39
    toGenerate = alphanumericData[var]
    ddpm = DDPM(1000, 256, 64, 64, 3, num_classes)
    device = "cpu"
    
    current_path = os.getcwd()
    dirname = os.path.dirname(current_path)
    modelPath = os.path.join(dirname, "project", "diffusion", "home","alphanumeric.pt")
    if exists(modelPath):
        ddpm.load_state_dict(load(modelPath, map_location = device))
    
    with no_grad():
        c = tensor([toGenerate] * 4).to(device)
        gen = ddpm.sample((4, 3, 32, 32), c, device)
        fig, ax = plt.subplots(2, 2, figsize = (4, 4))
        fig.set_facecolor("#0d193400")
        for i in range(4):
            img = gen[i].cpu().numpy()
            ax[i // 2, i % 2].axis("off")
            ax[i // 2, i % 2].imshow(img.transpose(1, 2, 0))
        file = f"{toGenerate}.png"
        finalPath = os.path.join(dirname, "project", "diffusion", "home", "static", file)
        print(finalPath)
        plt.savefig(finalPath)
        plt.close()
    return JsonResponse({'image_url':file}, safe=False)

def about(request):
    return render(request,'about.html')