import sys

sys.path.append("/mnt/Fedora2/code/python/tensorflow/rl/backend")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple
from CHESS.essentials.func import ChessHelper
from CHESS.model.batch import load_numpy
from tqdm import tqdm


class ChessPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(19, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self.policy_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 64 * 64)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        policy = self.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 8 * 8)
        policy = self.dropout(policy)
        policy = self.policy_fc(policy)
        policy = torch.softmax(policy, dim=1)

        return policy

    def save_model(self, path):
        torch.save(self.state_dict(), path)


def train(
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    save_path: str = "chess_policy_modelV2.pth",
):

    all_boards, all_evals = load_numpy()

    boards = np.concatenate(all_boards, axis=0)
    evals = np.concatenate(all_evals, axis=0)

    board_tensor = torch.from_numpy(boards).float()
    eval_tensor = torch.from_numpy(evals).float()

    dataset = torch.utils.data.TensorDataset(board_tensor, eval_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    model = ChessPolicy().cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{epochs}", total=len(dataloader)
        )

        for batch_boards, batch_evals in progress_bar:
            batch_boards = batch_boards.cuda()
            batch_evals = batch_evals.cuda()

            optimizer.zero_grad()

            policy_output = model(batch_boards)

            batch_evals = batch_evals.view(-1, 1)

            loss = criterion(policy_output, batch_evals)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        tqdm.write(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)

        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                save_path,
            )

    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        },
        save_path,
    )

    return model


def main() -> None:
    train()


if __name__ == "__main__":
    main()
