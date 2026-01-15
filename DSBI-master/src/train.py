import torch, os
from torch.utils.data import DataLoader
from dataset import BrailleDataset
from model import BddNet
from utils import visualize

def train(train_list, base_dir, save_path="checkpoints", epochs=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BddNet().to(device)
    dataset = BrailleDataset(train_list, base_dir)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    os.makedirs(save_path, exist_ok=True)
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for imgs, gts in loader:
            imgs, gts = imgs.to(device), gts.to(device)
            opt.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, gts)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{epochs}  Loss: {total_loss/len(loader):.5f}")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{save_path}/bddnet_epoch{epoch}.pt")
    print("Training complete.")

# inside train.py main section (at the bottom)
if __name__ == "__main__":
    train("../dummy_train.txt", "../data/", save_path="../checkpoints", epochs=100)
