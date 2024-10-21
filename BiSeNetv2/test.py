from PIL import Image
import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import torch.nn.functional as F

from dataset import KittiSemanticDataset, cityscapes_dataset
from visualization import KittiVisualizer
from utils.utils import preprocessing_cityscapes, preprocessing_kitti, postprocessing

from model.BiseNetv2 import BiSeNetV2
from model.ohem_loss import OhemCELoss

dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

def test(args):
    if args.dataset == 'kitti':
        dataset = KittiSemanticDataset()
    else:
        dataset = cityscapes_dataset()

    visualizer = KittiVisualizer()
    
    # define model
    model = BiSeNetV2(19)
    checkpoint = torch.load(args.weight_path, map_location=dev)
    model.load_state_dict(checkpoint['bisenetv2'], strict=False)
    model.eval()
    model.to(device)
    Loss = OhemCELoss(0.7)

    for i in range(len(dataset)):
        image, semantic = dataset[i]

        original = np.asarray(image.copy())

        # Preprocess the image
        if args.dataset == 'kitti':
            image = preprocessing_kitti(image)
        else:
            image = preprocessing_cityscapes(image)

        # Ensure the image is a 4D tensor: (batch_size, channels, height, width)
        image = image.unsqueeze(0).to(device)  # Add a batch dimension and move to device

        # Run the model
        pred = model(image)  # (1, 19, H, W)

        # loss from just logits (not including aux)
        loss = Loss(pred, torch.from_numpy(semantic).unsqueeze(0).to(device))  # Move semantic to device

        pred = postprocessing(pred)  # (H, W)

        # Coloring
        pred = visualizer.semantic_to_color(pred)  # (H, W, 3)

        # Get numpy image back
        image = image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)

        # Save results
        semantic = visualizer.semantic_to_color(semantic)  # (H, W, 3)
        visualizer.visualize_test(original, pred, semantic)

        if visualizer.pressed_btn == 27:
            cv2.destroyAllWindows()
            cv2.imwrite('./res.jpg', pred)
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='checkpoints/BiseNetv2_150.pth',)
    parser.add_argument('--dataset', choices=['cityscapes', 'kitti'], default='kitti')
    args = parser.parse_args()
    test(args)

