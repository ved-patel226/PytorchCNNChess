import sys

sys.path.append("/mnt/Fedora2/code/python/tensorflow/rl/backend")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from typing import List, Tuple
from CHESS.essentials.func import ChessHelper
from CHESS.DATA.loadNUMPY import load_numpy, get_files
from tqdm import tqdm
import os


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
        self.policy_fc = nn.Linear(32 * 8 * 8, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.tanh_scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        policy = self.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 8 * 8)
        policy = self.dropout(policy)
        policy = self.policy_fc(policy)

        policy = torch.tanh(policy) * self.tanh_scale
        return policy


class ChessDataset(IterableDataset):
    def __init__(self, data_dir="DATA", transform=None, batch_size=32):
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size

        self.boards, self.evals = self._get_files(data_dir)

        if len(self.boards) != len(self.evals):
            raise ValueError("Mismatch between board and evaluation files")

        self.total_samples = len(self.boards)
        self.current_index = 0

    def _get_files(self, data_dir):
        board_files = [
            f for f in os.listdir(data_dir) if "board" in f and f.endswith(".npy")
        ]
        eval_files = [
            f for f in os.listdir(data_dir) if "evals" in f and f.endswith(".npy")
        ]
        return board_files, eval_files

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= self.total_samples:
            self.current_index = 0
            raise StopIteration

        batch_boards = []
        batch_evals = []

        while (
            len(batch_boards) < self.batch_size
            and self.current_index < self.total_samples
        ):

            board_path = os.path.join(self.data_dir, self.boards[self.current_index])
            eval_path = os.path.join(self.data_dir, self.evals[self.current_index])

            board = np.load(board_path)
            eval_data = np.load(eval_path)

            board_tensor = torch.tensor(board, dtype=torch.float32)
            eval_tensor = torch.tensor(eval_data, dtype=torch.float32)

            if self.transform:
                board_tensor = self.transform(board_tensor)

            batch_boards.append(board_tensor)
            batch_evals.append(eval_tensor)

            del board, eval_data

            self.current_index += 1

        if batch_boards:
            return torch.stack(batch_boards), torch.stack(batch_evals)
        else:
            raise StopIteration

    def __len__(self):
        return self.total_samples


def loadDataset():
    dataset = ChessDataset(data_dir="DATA")
    dataloader = DataLoader(dataset, batch_size=None, num_workers=4, pin_memory=False)

    return dataloader


def train(
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    final_lr: float = 1e-6,
    save_path: str = "chess_policy_modelV4.pth",
):

    dataloader = loadDataset()

    model = ChessPolicy().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=final_lr
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{epochs}",
            total=84,
        )

        for batch_boards_total, batch_evals_total in progress_bar:
            for batch_board, batch_eval in zip(batch_boards_total, batch_evals_total):
                batch_boards = batch_board.cuda()
                batch_evals = batch_eval.cuda()

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
            scheduler.step()

        tqdm.write(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss}")

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            torch.save(
                {
                    "epoch": epoch,
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
