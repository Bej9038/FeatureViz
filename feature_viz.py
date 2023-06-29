"""
''' feature_viz.py '''

Description: Python script to visualize maximum class activation in intermediate layers in an ML model.
Author: Ben Jordan
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import models
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# model = models.vit_backbone()
model = models.resnet()


def main():
    model.to(device)
    model.eval()

    # num_features = model.fc.out_features
    num_features = 5
    lr = 0.01
    # wd = 1e-6
    wd = 0
    blur_every = 4
    iters = 50
    init_val = 1
    # Bx3x224x224 -> BxCxHxW
    # img = torch.tensor(np.full(shape=(num_features, 3, 224, 224), fill_value=init_val, dtype=np.float32), device=device,
    #                    requires_grad=True)
    img = torch.tensor(np.random.rand(num_features, 3, 224, 224), device=device,
                       requires_grad=True, dtype=torch.float)

    optimizer = optim.Adam([img], lr=lr, weight_decay=wd)
    loss_function = nn.CrossEntropyLoss()

    """ Set layers to identity function"""
    model.fc = models.Identity()
    model.meanpool = models.Identity()
    model.layer4 = models.Identity()
    # model.layer3 = models.Identity()
    # model.layer2 = models.Identity()
    # model.layer1 = models.Identity()
    # print(model)

    labels = torch.arange(start=0, end=num_features, device=device)
    blur = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=1.0).to(device)
    print("Starting image generation.")
    start = time.time()
    for k in range(iters):
        print(str(int(k/iters*100)) + "%")
        for i in range(blur_every):
            # print(str(i + 1) + "%")
            optimizer.zero_grad()
            output = model(img)
            loss = loss_function(output, labels)
            loss.backward()
            # print(loss.item())
            # if i % 2 == 0:
            #     norm = torch.norm(img.grad, dim=1)
            #     norm_vec = torch.flatten(norm)
            #     sorted_norm_vec = torch.sort(norm_vec)
            #     cutoff = int((len(norm_vec)-1) * 0.4)
            #     cutoff_val = sorted_norm_vec[0][cutoff]
            #     mask = norm[:, None, :, :] > cutoff_val
            #     img.grad *= mask
            #     img.grad *= 4

            optimizer.step()

        img = blur(img).detach()

        # norm = torch.norm(img, dim=1)[:, None, :, :]
        # norm_vec = torch.flatten(norm)
        # sorted_norm_vec = torch.sort(norm_vec)
        # cutoff = int((len(norm_vec)-1) * 0.1)
        # cutoff_val = sorted_norm_vec[0][cutoff]
        # cutoff_val = 0.01
        # mask = norm > cutoff_val
        # img *= mask

        img.requires_grad = True
        optimizer = optim.Adam([img], lr=lr, weight_decay=wd)

    img.requires_grad = False
    img = img.cpu()
    img -= torch.min(img) # set min value to 0
    mult = 0.86 / torch.max(img) # normalize values
    img *= mult
    for i in range(num_features):
        # print(torch.min(img[i]))
        plt.imshow(img[i].permute(1, 2, 0))
        # plt.savefig("../featureviz/full_model_feat" + str(i))
        plt.show()
    print("Images generation complete. Took " + str(round(time.time() - start, 2)) + " seconds.")


if __name__ == "__main__":
    main()
