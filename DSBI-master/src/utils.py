import cv2, numpy as np, math, matplotlib.pyplot as plt

def preprocess_image(img, size=(512, 512)):
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img

def parse_annotation(txt_path, img_shape):
    """
    Parse DSBI annotation file and return list of (x, y) dot centers.
    """
    h, w = img_shape
    with open(txt_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) < 4:
        return []

    angle = float(lines[0])
    verticals = np.array(list(map(float, lines[1].split())))
    horizontals = np.array(list(map(float, lines[2].split())))

    coords = []
    for cell in lines[3:]:
        vals = list(map(int, cell.split()))
        if len(vals) != 8: 
            continue
        row, col = vals[0], vals[1]
        dots = vals[2:]

        # Protect against index overflow
        if 2*(col-1)+1 >= len(verticals) or 3*(row-1)+3 >= len(horizontals):
            continue

        x0, x1 = verticals[2*(col-1)], verticals[2*(col-1)+1]
        y0, y3 = horizontals[3*(row-1)], horizontals[3*(row-1)+3]
        dx, dy = (x1 - x0)/2, (y3 - y0)/3

        # 6 standard Braille dot centers (x, y)
        centers = [
            (x0 + dx/2, y0 + dy/2),       # dot 1
            (x0 + dx/2, y0 + 1.5*dy),     # dot 2
            (x0 + dx/2, y0 + 2.5*dy),     # dot 3
            (x0 + 1.5*dx, y0 + dy/2),     # dot 4
            (x0 + 1.5*dx, y0 + 1.5*dy),   # dot 5
            (x0 + 1.5*dx, y0 + 2.5*dy),   # dot 6
        ]

        for i in range(6):
            if dots[i] == 1:
                coords.append(centers[i])

    # Ignore rotation because images are deskewed already
    return coords

def generate_gaussian_map(shape, coords, sigma=2):
    h, w = shape
    yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    heatmap = np.zeros((h, w), dtype=np.float32)
    for (x, y) in coords:
        heatmap += np.exp(-((xv - y)**2 + (yv - x)**2) / (2 * sigma**2))
    heatmap = np.clip(heatmap, 0, 1)
    return heatmap

def visualize(img, gt, pred=None):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img, cmap='gray'); plt.title("Input")
    plt.subplot(1,3,2); plt.imshow(gt, cmap='hot'); plt.title("Ground Truth")
    if pred is not None:
        plt.subplot(1,3,3); plt.imshow(pred, cmap='hot'); plt.title("Prediction")
    plt.show()
