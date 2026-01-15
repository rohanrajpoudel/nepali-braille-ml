import cv2, torch, numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def extract_dots(pred_map, thresh=0.5):
    mask = (pred_map > thresh).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    return centroids[1:]  # skip background

def compare_dots(pred, gt, dist_thresh=5):
    tp = fp = fn = 0
    gt_used = np.zeros(len(gt), dtype=bool)
    for p in pred:
        dists = np.linalg.norm(gt - p, axis=1)
        min_i = np.argmin(dists)
        if dists[min_i] < dist_thresh and not gt_used[min_i]:
            tp += 1; gt_used[min_i] = True
        else:
            fp += 1
    fn = len(gt) - tp
    return tp, fp, fn

def evaluate_model(model, val_loader, device="cpu"):
    model.eval()
    TP=FP=FN=0
    with torch.no_grad():
        for imgs, gts in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs).cpu().numpy()
            for i in range(len(preds)):
                pred_map = preds[i,0]
                gt_map = gts[i,0].numpy()
                gt_coords = extract_dots(gt_map, 0.5)
                pr_coords = extract_dots(pred_map, 0.5)
                tp, fp, fn = compare_dots(pr_coords, gt_coords)
                TP+=tp; FP+=fp; FN+=fn
    precision = TP/(TP+FP+1e-6)
    recall = TP/(TP+FN+1e-6)
    f1 = 2*precision*recall/(precision+recall+1e-6)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
