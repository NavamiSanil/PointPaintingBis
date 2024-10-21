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
    
    # Define model
    model = BiSeNetV2(19)
    checkpoint = torch.load(args.weight_path, map_location=dev, weights_only=True)  # Suppress FutureWarning
    model.load_state_dict(checkpoint['bisenetv2'], strict=False)
    model.eval()
    model.to(device)
    Loss = OhemCELoss(0.7)

    for i in range(len(dataset)):
        image, semantic = dataset[i]
        original = np.asarray(image.copy())

        # Preprocess the image based on the dataset type
        image = preprocessing_kitti(image) if args.dataset == 'kitti' else preprocessing_cityscapes(image)

        # Add batch dimension and move to device
        image = image.unsqueeze(0).to(device)  # (1, 3, H, W)

        # Get model predictions
        with torch.no_grad():
            pred = model(image)  # (1, 19, H, W)
        
        # Calculate loss (make sure semantic is on the same device as pred)
        semantic_tensor = torch.from_numpy(semantic).unsqueeze(0).to(device)  # Move to the same device
        loss = Loss(pred, semantic_tensor)
        print(f'Loss for sample {i}: {loss.item()}')

        pred = postprocessing(pred)  # (H, W) 
        pred = visualizer.semantic_to_color(pred)  # (H, W, 3) - color mapping

        # Save and visualize
        visualizer.visualize_test(original, pred, semantic)
        
        # Optional: handle exit on key press
        if visualizer.pressed_btn == 27:
            cv2.destroyAllWindows()
            cv2.imwrite('./res.jpg', pred)
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='checkpoints/BiseNetv2_150.pth')
    parser.add_argument('--dataset', choices=['cityscapes', 'kitti'], default='kitti')
    args = parser.parse_args()
    test(args)
