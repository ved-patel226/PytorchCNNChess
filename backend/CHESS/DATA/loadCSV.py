import sys

sys.path.append("/mnt/Fedora2/code/python/tensorflow/rl/backend")

import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import csv
import os
import numpy as np
from io import StringIO
from CHESS.essentials.func import ChessHelper
import chess
from typing import List, Tuple
import h5py
from tqdm import tqdm
import datetime
import glob


def get_line_offsets(file_path: str, num_processes: int) -> Tuple[List[int], int]:
    with open(file_path, "r") as f:
        total_lines = sum(1 for _ in f)
        lines_per_process = total_lines // num_processes
        f.seek(0)
        offsets = [0]
        current_line = 0

        for _ in range(num_processes - 1):
            target_line = current_line + lines_per_process
            while current_line < target_line:
                f.readline()
                current_line += 1
            offsets.append(f.tell())

        offsets.append(os.path.getsize(file_path))
        return offsets, total_lines


def count_lines_in_chunk(file_path: str, start_offset: int, end_offset: int) -> int:
    count = 0
    with open(file_path, "r") as f:
        f.seek(start_offset)
        if start_offset != 0:
            f.readline()
        while f.tell() < end_offset:
            f.readline()
            count += 1
    return count


def process_and_save_chunk(
    chunk_data, output_file, chunk_num, min_eval=None, max_eval=None
):
    if not chunk_data:
        return

    boards = np.array([item[0] for item in chunk_data])

    evals = np.array(
        [
            normalize_eval(
                item[1],
                min_val=min_eval,
                max_val=max_eval,
            )
            for item in chunk_data
        ],
        dtype=np.float32,
    )

    np.save(
        f"DATA/{output_file}_boards_{chunk_num}",
        boards,
    )
    np.save(
        f"DATA/{output_file}_evals_{chunk_num}",
        evals,
    )

    del boards
    del evals


def process_file_chunk(args):
    (
        file_path,
        start_offset,
        end_offset,
        chess_helper,
        chunk_size,
        process_id,
        output_file,
    ) = args

    total_lines = count_lines_in_chunk(file_path, start_offset, end_offset)

    position = process_id + 1
    pbar = tqdm(
        total=total_lines, desc=f"Process {process_id}", position=position, leave=False
    )

    current_chunk = []
    chunk_counter = 0

    with open(file_path, "r") as f:
        f.seek(start_offset)
        if start_offset != 0:
            f.readline()

        current_position = f.tell()

        min_eval = float("inf")
        max_eval = float("-inf")

        while current_position < end_offset:
            line = f.readline()
            current_position = f.tell()

            if not line.strip():
                pbar.update(1)
                continue

            try:
                row = next(csv.reader([line]))

                # Validate row has enough columns
                if len(row) < 2:
                    tqdm.write(f"Skipping malformed row: {line}")
                    pbar.update(1)
                    continue

                # Attempt to create board
                try:
                    board = chess_helper.tokenize(board=chess.Board(row[0]))
                except ValueError as e:
                    tqdm.write(f"Invalid FEN: {row[0]} - {e}")
                    pbar.update(1)
                    continue

                # Handle evaluation value
                eval_str = row[1].replace("#", "")
                try:
                    row_as_float = float(eval_str)
                except ValueError:
                    tqdm.write(f"Could not convert evaluation to float: {row[1]}")
                    pbar.update(1)
                    continue

                # Multiply by 2 if it was a checkmate evaluation
                if "#" in row[1]:
                    row_as_float *= 2

                # Track min and max evaluations
                min_eval = min(min_eval, row_as_float)
                max_eval = max(max_eval, row_as_float)

                current_chunk.append([board, row_as_float])

                if len(current_chunk) >= chunk_size:
                    process_and_save_chunk(
                        current_chunk,
                        output_file,
                        f"{process_id}_{chunk_counter}",
                        min_eval=min_eval,
                        max_eval=max_eval,
                    )
                    chunk_counter += 1
                    current_chunk = []

                pbar.update(1)

            except Exception as e:
                tqdm.write(f"Unexpected error processing line: {line.strip()} - {e}")
                pbar.update(1)

    # Process any remaining chunks
    if current_chunk:
        process_and_save_chunk(
            current_chunk,
            output_file,
            f"{process_id}_{chunk_counter}",
            min_eval=min_eval,
            max_eval=max_eval,
        )

    pbar.close()
    return chunk_counter + 1


def normalize_eval(value, min_val=-54000, max_val=43000, target_min=-10, target_max=10):

    clipped_value = max(min(value, max_val), min_val)
    normalized = ((clipped_value - min_val) / (max_val - min_val)) * (
        target_max - target_min
    ) + target_min
    return normalized


def load_csv(file_path: str, output_file: str, num_processes: int = None) -> None:
    if num_processes is None:
        num_processes = mp.cpu_count()

    chess_helper = ChessHelper()
    offsets, total_lines = get_line_offsets(file_path, num_processes)
    chunk_size = 3911
    # chunk_size = 100

    process_args = []
    for i in range(num_processes):
        process_args.append(
            (
                file_path,
                offsets[i],
                offsets[i + 1],
                chess_helper,
                chunk_size,
                i,
                output_file,
            )
        )

    main_pbar = tqdm(total=total_lines, desc="Total Progress", position=0)

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for _ in executor.map(process_file_chunk, process_args):
            main_pbar.update(chunk_size)

    main_pbar.close()

    print("\n" * (num_processes + 1))


def main() -> None:
    csv_file = "archive/tactic_evals.csv"
    file = "chess_data"

    os.makedirs("DATA", exist_ok=True)

    print(f"Processing CSV file {csv_file}")
    load_csv(csv_file, file)
    print(f"Data saved to {file}")


if __name__ == "__main__":
    main()
