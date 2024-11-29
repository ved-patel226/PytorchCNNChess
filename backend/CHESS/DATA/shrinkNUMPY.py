import numpy as np
import glob
from tqdm import tqdm

board_files = sorted(glob.glob("DATAPIECES/chess_data_boards_*.npy"))[:-25]
eval_files = sorted(glob.glob("DATAPIECES/chess_data_evals_*.npy"))[:-25]


boards = []
for f in tqdm(board_files, desc="Loading board files"):
    boards.append(np.load(f))

all_boards = np.concatenate(boards)
np.save("combined_boards.npy", all_boards)

del boards, all_boards

evals = []
for f in tqdm(eval_files, desc="Loading eval files"):
    evals.append(np.load(f))
all_evals = np.concatenate(evals)
np.save("combined_evals.npy", all_evals)
