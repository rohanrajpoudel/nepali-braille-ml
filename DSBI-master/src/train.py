import torch, os
from torch.utils.data import DataLoader
from dataset import BrailleDataset
from model import BddNet
from utils import visualize

def train(train_list, base_dir, save_path="checkpoints", epochs=100):
    print(">>> Starting training function")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(">>> Device:", device)

    model = BddNet().to(device)
    print(">>> Model loaded")

    dataset = BrailleDataset(train_list, base_dir)
    print(">>> Dataset size:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,      # IMPORTANT (Windows)
        pin_memory=False
    )
    print(">>> DataLoader ready")

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        print(f">>> Epoch {epoch} started")
        model.train()

        for i, (imgs, gts) in enumerate(loader):
            print(f"   Batch {i} loaded")

            imgs, gts = imgs.to(device), gts.to(device)

            opt.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, gts)
            loss.backward()
            opt.step()

            print(f"   Batch {i} loss: {loss.item():.6f}")

        print(f">>> Epoch {epoch} finished")

# inside train.py main section (at the bottom)
if __name__ == "__main__":
    train("../dummy_train.txt", "../data/", save_path="../checkpoints", epochs=100)
