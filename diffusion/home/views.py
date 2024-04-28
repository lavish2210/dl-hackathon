from django.shortcuts import render
from django.http import JsonResponse 
from .GenModel import DDPM
from .GenModelAnimal import DDPM as DDPM_animal
from os.path import exists
from torch import tensor, no_grad, load
import matplotlib.pyplot as plt
import matplotlib
import os

device = "cuda"

alphanumericData = {'#': 0,
    '$': 1,
    '&': 2,
    "O": 3,
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

animalData = {
 'dog': 0,
 'horse': 1,
 'elephant': 2,
 'butterfly': 3,
 'chicken': 4,
 'cat': 5,
 'cow': 6,
 'sheep': 7,
 'spider': 8,
 'squirrel':9
}

ddpm_alnum = DDPM(1000, 256, 64, 64, 3, 39).to(device)
is_alnum_loaded = False
ddpm_animals = DDPM_animal(1000, 512, 128, 128, 3, 10).to(device)
is_animal_loaded = False

def home(request): 
    return render(request, 'home.html')

def get_image(request):
    global is_animal_loaded, is_alnum_loaded, ddpm_alnum, ddpm_animals
    matplotlib.use('agg')
    var = request.GET.get('input_text')

    if var in alphanumericData.keys():

        if (var >= 'a' or var <= 'z'):
            var = var.upper()
        
        num_classes = 39
        toGenerate = alphanumericData[var]
        device = "cuda"
        
        current_path = os.getcwd()
        dirname = os.path.dirname(current_path)
        modelPath = os.path.join(dirname, "diffusion", "home","alphanumeric.pt")
        if exists(modelPath) and not is_alnum_loaded:
            ddpm_alnum.load_state_dict(load(modelPath, map_location = device))
            is_alnum_loaded = True
            print("loaded model")
        
        with no_grad():
            c = tensor([toGenerate] * 9).to(device)
            gen = ddpm_alnum.sample((9, 3, 32, 32), c, device)
            fig, ax = plt.subplots(3, 3, figsize = (4, 4))
            fig.set_facecolor("#0d193400")
            for i in range(9):
                img = gen[i].cpu().numpy()
                ax[i // 3, i % 3].axis("off")
                ax[i // 3, i % 3].imshow(img.transpose(1, 2, 0))
            file = f"{toGenerate}.png"
            finalPath = os.path.join(dirname, "diffusion", "home", "static", file)
            print(finalPath)
            plt.savefig(finalPath)
            plt.close()
    elif var in animalData.keys():
        var = var.lower()
        
        num_classes = 10
        toGenerate = animalData[var]
        device = "cuda"
        
        current_path = os.getcwd()
        dirname = os.path.dirname(current_path)
        modelPath = os.path.join(dirname, "diffusion", "home","animals.pt")
        if exists(modelPath) and not is_animal_loaded:
            ddpm_animals.load_state_dict(load(modelPath, map_location = device))
            is_animal_loaded = True
            print("loaded model")
        
        with no_grad():
            c = tensor([toGenerate] * 2).to(device)
            gen = ddpm_animals.sample((2, 3, 128, 128), c, device)
            fig, ax = plt.subplots(2, 1, figsize = (4, 4))
            fig.set_facecolor("#0d193400")
            for i in range(2):
                img = gen[i].cpu().numpy()
                ax[i].axis("off")
                ax[i].imshow(img.transpose(1, 2, 0))
            file = f"{toGenerate}.png"
            finalPath = os.path.join(dirname, "diffusion", "home", "static", file)
            print(finalPath)
            plt.savefig(finalPath)
            plt.close()
    return JsonResponse({'image_url':file}, safe=False)

def about(request):
    return render(request,'about.html')