from math import log, sqrt
from time import time
from os.path import exists
from multiprocessing import Pool

from torch.optim import Adam
from torch import cat, tensor, randn, zeros, sin, cos, arange, exp, randint, square, sqrt as square_root, no_grad, min as minimum, max as maximum, ones, save, load
from torch.utils.data import DataLoader


# In[2]:


from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize, RandomHorizontalFlip


# In[3]:


import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, FFMpegWriter

def train(epochs, opt, ddpm, loader, valid_loader, time_steps, sample_every, num_classes):
    min_loss = 10000
    fixed_c = randint(0, num_classes - 1, (4,)).to(device)
    for epoch in range(epochs):
        with tqdm(loader, unit = "batch") as tqdm_loader:
            for x, c in tqdm_loader:
                tqdm_loader.set_description(f"Training {epoch + 1}")
                opt.zero_grad(set_to_none = True)
                x = x.to(device)
                c = c.to(device)
                t = randint(time_steps, (x.shape[0],)).to(device)
                eps, noise = ddpm(x, c, t, device)
                loss = square(eps - noise).mean()
                loss.backward()
                opt.step()
                tqdm_loader.set_postfix(loss = loss.item())
        with no_grad():
            if (epoch + 1) % sample_every == 0:
                print ("Sampling images")
                for j in range(num_classes):
                    c = tensor([j] * 4).to(device)
                    gen = ddpm.sample((4, 3, 128, 128), c, device)
                    fig, ax = plt.subplots(2, 2)
                    for i in range(4):
                        img = gen[i].cpu().numpy()
                        ax[i // 2, i % 2].axis("off")
                        ax[i // 2, i % 2].imshow(img.transpose(1, 2, 0))
                    plt.savefig(f"animals_generated/Generated_animals_{epoch + 1}_{j}.png")
                    plt.close()
        save(ddpm.state_dict(), "animals.pt")


# In[20]:
if __name__ == "__main__":
    dataset = ImageFolder("raw-img/train", Compose([ToTensor(), Resize((128, 128)), RandomHorizontalFlip(), Normalize()]))
    num_classes = len(dataset.find_classes("raw-img/train")[0])
    train_loader = DataLoader(dataset, shuffle = True, batch_size = 4)

    device = "cuda:0"


    # In[21]:

    ddpm = DDPM(1000, 512, 128, 128, 3, num_classes).to(device)
    if exists("animals.pt"):
        ddpm.load_state_dict(load("animals.pt"))
    opt = Adam(ddpm.parameters(), lr = 1e-5)


    # In[22]:


    imgs, _ = next(iter(train_loader))
    fig, ax = plt.subplots(2, 2)
    for i in range(4):
        img = imgs[i].cpu().numpy()
        img = (img + 1.0) / 2.0
        ax[i // 2, i % 2].axis("off")
        ax[i // 2, i % 2].imshow(img.transpose(1, 2, 0))
    plt.savefig("animals_generated/First_Batch_animals.png")
    plt.close()


    # In[ ]:


    train(20, opt, ddpm, train_loader, None, 1000, 1, num_classes)