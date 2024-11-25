import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_numpy() -> tuple[np.ndarray, np.ndarray]:
    formatted_data_dir = "FORMATTED_DATA"
    files = [
        name
        for name in os.listdir(formatted_data_dir)
        if os.path.isfile(os.path.join(formatted_data_dir, name))
    ]

    assert len(files) == 2, f"Expected 2 files, found {files}"

    board_file = [f for f in files if "boards" in f][0]
    eval_file = [f for f in files if "evals" in f][0]

    board = np.load(os.path.join(formatted_data_dir, board_file))
    evals = np.load(os.path.join(formatted_data_dir, eval_file))

    return board, evals


def main() -> None:
    board, evals = load_numpy()

    batches = []

    for i in range(0, len(board), 773):
        batch_boards = board[i : i + 773]
        batch_evals = evals[i : i + 773]

        batches.append((batch_boards, batch_evals))

    print(len(batches))


if __name__ == "__main__":
    main()
