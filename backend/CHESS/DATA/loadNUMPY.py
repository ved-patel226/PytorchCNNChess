import numpy as np
import os
import torch


def get_files(formatted_data_dir="DATA"):

    files = [
        name
        for name in os.listdir(formatted_data_dir)
        if os.path.isfile(os.path.join(formatted_data_dir, name))
    ]

    evals = [file for file in files if "evals" in file]
    boards = [file for file in files if "boards" in file]

    print("LOADED NEW FILES")

    return boards, evals, formatted_data_dir


def load_numpy(prevBoardIndex=None) -> tuple[np.ndarray, np.ndarray, int]:
    boards, evals, formatted_data_dir = get_files()

    if prevBoardIndex is None:
        index = 0
    else:
        index = prevBoardIndex + 1

    if index >= len(boards):
        index = len(boards) - 1

    board = np.load(os.path.join(formatted_data_dir, boards[index]))
    eval = np.load(os.path.join(formatted_data_dir, evals[index]))

    return board, eval


def main() -> None:
    board, _, _ = get_files()

    for i in range((len(board) * 2)):
        board, _ = load_numpy(i)
        print(board.shape)


if __name__ == "__main__":
    main()
