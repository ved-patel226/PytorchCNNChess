import sys

sys.path.append("/mnt/Fedora2/code/python/tensorflow/rl")

from CHESS.DATA.loadNUMPY import load_numpy


def getDATA():
    board, evals = load_numpy()

    data = []
    for i in range(0, len(board), 773):
        batch_boards = board[i : i + 773]
        batch_evals = evals[i : i + 773]

        data.append((batch_boards, batch_evals))

    return data


def main() -> None:
    data = getDATA()


if __name__ == "__main__":
    main()
