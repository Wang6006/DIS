import os
import time
import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm

import torch, gc
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from models import *


if __name__ == "__main__":
    dataset_path = "/content/drive/MyDrive/NLCN/Dataset/my-dataset"
    model_path = "/content/drive/MyDrive/NLCN/ISNet/ISNet-Model.pth"
    result_path = "/content/drive/MyDrive/NLCN/Results/my-results"
    input_size = [1024, 1024]

    os.makedirs(result_path, exist_ok=True)

    net = ISNetDIS()
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))

    net.eval()
    im_list = glob(dataset_path + "/*.jpg") + \
              glob(dataset_path + "/*.JPG") + \
              glob(dataset_path + "/*.jpeg") + \
              glob(dataset_path + "/*.JPEG") + \
              glob(dataset_path + "/*.png") + \
              glob(dataset_path + "/*.PNG") + \
              glob(dataset_path + "/*.bmp") + \
              glob(dataset_path + "/*.BMP") + \
              glob(dataset_path + "/*.tiff") + \
              glob(dataset_path + "/*.TIFF")

    with torch.no_grad():
        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
            print("im_path: ", im_path)
            im = io.imread(im_path)
            if len(im.shape) < 3:
                im = np.repeat(im[:, :, np.newaxis], 3, axis=2)  # Convert grayscale to RGB
            im_shp = im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
            im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear", align_corners=False).type(torch.uint8)
            image = torch.divide(im_tensor, 255.0)
            image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

            if torch.cuda.is_available():
                image = image.cuda()

            result = net(image)
            result = torch.squeeze(F.interpolate(result[0][0], im_shp, mode='bilinear', align_corners=False), 0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result - mi) / (ma - mi + 1e-8)  # avoid division by zero

            im_name = os.path.splitext(os.path.basename(im_path))[0]

            # === Save result ===
            result_np = (result * 255).cpu().numpy().astype(np.uint8)

            # If result is (H, W), save directly. If (1, H, W), squeeze first.
            if result_np.ndim == 3 and result_np.shape[0] == 1:
                result_np = result_np.squeeze(0)

            save_path = os.path.join(result_path, im_name + ".png")
            io.imsave(save_path, result_np)

