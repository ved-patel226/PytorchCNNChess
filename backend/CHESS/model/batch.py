import sys

sys.path.append("/mnt/Fedora2/code/python/tensorflow/rl")

from CHESS.DATA.loadNUMPY import load_numpy
from tqdm import tqdm


def getDATA():
    board, evals = load_numpy()

    all_boards = []
    all_evals = []

    for i in tqdm(range(0, len(board), 773)):
        batch_boards = board[i : i + 773]
        batch_evals = evals[i : i + 773]

        all_boards.append(batch_boards)
        all_evals.append(batch_evals)

    return all_boards, all_evals


def main() -> None:
    data = getDATA()


if __name__ == "__main__":
    main()
