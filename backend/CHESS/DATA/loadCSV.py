from tqdm import tqdm
import threading
import csv
from queue import Queue
import os
import numpy as np
from io import StringIO
from CHESS.essentials.func import ChessHelper
import chess
from typing import List, Tuple
import h5py
import datetime
import glob


def get_line_offsets(file_path: str, num_threads: int) -> Tuple[List[int], int]:
    with open(file_path, "r") as f:
        total_lines = sum(1 for _ in f)
        lines_per_thread = total_lines // num_threads
        f.seek(0)
        offsets = [0]
        current_line = 0

        for _ in range(num_threads - 1):
            target_line = current_line + lines_per_thread
            while current_line < target_line:
                f.readline()
                current_line += 1
            offsets.append(f.tell())

        offsets.append(os.path.getsize(file_path))
        return offsets, total_lines


def load_from_h5(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load data from HDF5 file"""
    with h5py.File(file_path, "r") as f:
        boards = f["boards"][:]
        evals = f["evals"][:]
        return boards, evals


def load_csv(file_path: str, output_file: str, num_threads: int = None) -> None:
    """Load CSV and save to numpy arrays in smaller chunks"""
    if num_threads is None:
        num_threads = max(1, os.cpu_count() - 2)

    offsets, total_lines = get_line_offsets(file_path, num_threads)
    threads = []
    result_queue = Queue(maxsize=100)
    chess_helper = ChessHelper()

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    chunk_counter = 0

    def save_chunk(chunk_data, chunk_num):
        """Helper function to save a single chunk"""
        if not chunk_data:
            return

        boards = np.array([item[0] for item in chunk_data])
        evals = np.array(
            [
                (
                    float(item[1])
                    if "#" not in item[1]
                    else float(item[1].replace("#", "")) * 1000
                )
                for item in chunk_data
            ],
            dtype=np.float32,
        )

        np.save(f"FORMATTED_DATA/{output_file}_boards_{chunk_num}", boards)
        np.save(f"FORMATTED_DATA/{output_file}_evals_{chunk_num}", evals)

        del boards
        del evals

    with tqdm(total=total_lines, desc="Total Progress", position=0, leave=True) as pbar:
        # Start threads
        for i in range(num_threads):
            thread = threading.Thread(
                target=read_chunk,
                args=(
                    file_path,
                    offsets[i],
                    offsets[i + 1],
                    result_queue,
                    i + 1,
                    pbar,
                    chess_helper,
                    500,  # Reduced chunk size
                ),
            )
            threads.append(thread)
            thread.start()

        current_chunk = []
        active_threads = len(threads)
        items_in_current_chunk = 0
        max_chunk_size = 25_000

        while active_threads > 0 or not result_queue.empty():
            try:
                # Shorter timeout to check thread status more frequently
                chunk = result_queue.get(timeout=0.1)

                # Process each item individually to control memory better
                for item in chunk:
                    current_chunk.append(item)
                    items_in_current_chunk += 1

                    # Save when reach max chunk size
                    if items_in_current_chunk >= max_chunk_size:
                        save_chunk(current_chunk, chunk_counter)
                        chunk_counter += 1
                        current_chunk = []
                        items_in_current_chunk = 0

                # Clear the processed chunk to free memory
                del chunk

            except Exception as e:
                # Check if any threads have finished
                active_threads = sum(1 for thread in threads if thread.is_alive())
                continue

            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue

        if current_chunk:
            save_chunk(current_chunk, chunk_counter)

        for thread in threads:
            thread.join()


def read_chunk(
    file_path: str,
    start_offset: int,
    end_offset: int,
    result_queue: Queue,
    thread_id: int,
    pbar: tqdm,
    chess_helper: ChessHelper,
    chunk_size: int = 500,
) -> None:
    current_chunk = []
    with open(file_path, "r") as f:
        f.seek(start_offset)
        if start_offset != 0:
            f.readline()

        chunk_content = StringIO(f.read(end_offset - f.tell()))
        line_count = sum(1 for _ in chunk_content)
        chunk_content.seek(0)

        with tqdm(
            total=line_count,
            desc=f"Thread {thread_id}",
            position=thread_id,
            leave=False,
        ) as pbar2:
            for line in chunk_content:
                if line.strip():
                    try:
                        row = next(csv.reader([line]))
                        try:
                            board = chess_helper.tokenize(board=chess.Board(row[0]))
                            current_chunk.append([board, row[1]])

                            if len(current_chunk) >= chunk_size:
                                result_queue.put(current_chunk)
                                current_chunk = []

                        except Exception as e:
                            continue
                        pbar2.update(1)
                        pbar.update(1)
                    except StopIteration:
                        continue

            if current_chunk:
                result_queue.put(current_chunk)

            pbar2.clear()
            pbar2.close()


def main() -> None:
    csv_file = "DATA/archive/random_evals.csv"
    h5_file = "chess_data"

    print(f"Processing CSV file {csv_file}")
    load_csv(csv_file, h5_file)
    print(f"Data saved to {h5_file}")


if __name__ == "__main__":
    main()
