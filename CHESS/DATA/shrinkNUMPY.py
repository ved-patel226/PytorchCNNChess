import os
import numpy as np

data_dir = "FORMATTED_DATA"
NPY_BOARDS = [f for f in os.listdir(data_dir) if f.endswith(".npy") and "boards" in f]
NPY_EVALS = [f for f in os.listdir(data_dir) if f.endswith(".npy") and "evals" in f]
print(f"Found {len(NPY_BOARDS)} board files and {len(NPY_EVALS)} eval files")
assert len(NPY_BOARDS) == len(NPY_EVALS)

all_boards = []
all_evals = []

for board_file, eval_file in zip(NPY_BOARDS, NPY_EVALS):
    board = np.load(os.path.join(data_dir, board_file))
    evals = np.load(os.path.join(data_dir, eval_file))

    all_boards.append(board)
    all_evals.append(evals)

combined_boards = np.concatenate(all_boards, axis=0)
combined_evals = np.concatenate(all_evals, axis=0)

print(f"Combined boards shape: {combined_boards.shape}")
print(f"Combined evals shape: {combined_evals.shape}")

np.save(os.path.join(data_dir, "all_evals.npy"), combined_evals)
np.save(os.path.join(data_dir, "all_boards.npy"), combined_boards)
