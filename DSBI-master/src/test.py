import torch, cv2, numpy as np, os
from model import BddNet
from utils import preprocess_image, visualize

def infer_single(image_path, checkpoint, out_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BddNet().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    inp = preprocess_image(img)
    inp = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred = model(inp)[0,0].cpu().numpy()
    visualize(img, np.zeros_like(img), pred)
    if out_path:
        cv2.imwrite(out_path, (pred*255).astype(np.uint8))
    return pred
