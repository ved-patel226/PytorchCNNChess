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

    batch_size = 773
    num_batches = len(board) // batch_size
    board = board[: num_batches * batch_size].reshape(-1, batch_size, *board.shape[1:])
    evals = evals[: num_batches * batch_size].reshape(-1, batch_size, *evals.shape[1:])

    return board, evals


def main() -> None:
    board, evals = load_numpy()

    print(board.shape, evals.shape)


if __name__ == "__main__":
    main()
