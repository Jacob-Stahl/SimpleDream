import torch
import torchvision
from torch import optim
import numpy as np
import PIL
import os

def output_to_image(out_img):

    out_img = out_img.squeeze_()
    out_img = out_img.cpu()
    out_img = out_img.detach().numpy()
    out_img = np.uint8(out_img*255)
    out_img = np.swapaxes(out_img, 0, 2)
    out_img = PIL.Image.fromarray(out_img)
    out_img = out_img.rotate(270, expand = True)
    out_img = out_img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    out = out_img
    del out_img
    torch.cuda.empty_cache()
    return out

def image_to_input(img):

    img = np.asarray(img, dtype= np.float32)
    img = np.transpose(img, (2,0,1))
    img = img / 255
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    img = img.to(device)

    out = img
    del img
    torch.cuda.empty_cache()
    return out

def deep_dream(image, model, num_iterations, depth, lr, gamma):

    image = torch.autograd.Variable(image, requires_grad=True)
    original = torch.autograd.Variable(image, requires_grad=True)

    layers = list(model.features.modules())
    model.zero_grad()
    
    optimizer = optim.Adam([image])
    for i in range(num_iterations):
        output_encoding = image
        for j in range(1,depth):
            layer = layers[j]
            output_encoding = layer(output_encoding)

        norm = output_encoding.norm() ** (1/2)
        diff = (image - original)
        if i % 10 == 0:
            print("layer norm : ", norm.item(), end = "   ")
            print("image diff : ", diff.norm().item())

        loss = diff.norm() ** 2 - norm
        loss.backward()
        image.grad.data = torch.clamp(image.grad.data, -lr, lr) 
        optimizer.step()

    return torch.tensor(image.clone().detach(), requires_grad= False)

if __name__ == "__main__":

    model = torchvision.models.vgg11(pretrained=True, progress=True)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    model.to(device)

    input_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input_images/')
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_images/')
    input_images = os.listdir(input_folder)

    print("pulling images from : ", input_folder)
    print("dumping to          : ", output_folder)
    print("counting sheep...")

    for image in input_images:
        
        print("     ", image)
        img = PIL.Image.open(os.path.join(input_folder, image))
        model_output = image_to_input(img)
        for i in range (3):
            model_output = deep_dream(image = model_output,model = model, num_iterations = 25, depth = 16, lr = 0.05, gamma = .2)
            model_output = deep_dream(image = model_output,model = model, num_iterations = 25, depth = 21, lr = 0.05, gamma = .2)
        dream_image = output_to_image(model_output)

        del model_output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dream_image.save(os.path.join(output_folder, "dream_" + image))