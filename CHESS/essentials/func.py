import chess
from pprint import pprint
import numpy as np
import time
from typing import Tuple


class ChessHelper:
    CHESS_PIECES = ["p", "n", "b", "q", "k", "r"]
    FILES = ["a", "b", "c", "d", "e", "f", "g", "h"]
    RANKS = [str(i) for i in range(1, 9)]
    TOKENS = {}

    def __init__(self):
        self.board = chess.Board()

        self.createTokens()

    def timeit(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
            return result

        return wrapper

    def getBoard(self, debug=False):
        board_2d = []
        for rank in chess.RANK_NAMES:
            row = []
            for file in chess.FILE_NAMES:
                square = chess.square(
                    chess.FILE_NAMES.index(file), chess.RANK_NAMES.index(rank)
                )
                row.append(str(self.board.piece_at(square) or "."))
            board_2d.append(row)

        pprint(board_2d) if debug else None
        return board_2d

    def boardToMatrix(self, debug=False):
        master_board = []

        for i in range(1, 5):
            if i == 4:
                master_board.append(self.getBoard())
                break

            for piece in self.CHESS_PIECES:
                BIGrow = []
                for rank in chess.RANK_NAMES:
                    SMALLrow = []
                    for file in chess.FILE_NAMES:
                        square = chess.square(
                            chess.FILE_NAMES.index(file), chess.RANK_NAMES.index(rank)
                        )

                        piece_at_square = self.board.piece_at(square)

                        if i == 1:
                            SMALLrow.append(
                                piece_at_square.symbol()
                                if piece_at_square
                                and piece_at_square.symbol() == piece.lower()
                                else "."
                            )
                        elif i == 2:
                            SMALLrow.append(
                                piece_at_square.symbol()
                                if piece_at_square
                                and piece_at_square.symbol() == piece.upper()
                                else "."
                            )
                        elif i == 3:
                            SMALLrow.append(
                                piece_at_square.symbol()
                                if piece_at_square
                                and piece_at_square.symbol().upper() == piece.upper()
                                else "."
                            )

                    BIGrow.append(SMALLrow)
                master_board.append(BIGrow)

        pprint(master_board) if debug else None
        print("Length of master_board: ", len(master_board)) if debug else None

        return master_board

    def createTokens(self):
        self.TOKENS["."] = 0

        for i in self.CHESS_PIECES:
            self.TOKENS[i] = len(self.TOKENS)
            self.TOKENS[i.upper()] = len(self.TOKENS)

        # print(self.TOKENS)

    def tokenize(self, board=None, debug=False):
        if board == None:
            board = self.board

        boardMatrix = self.boardToMatrix()

        for board in boardMatrix:
            for row in board:
                for piece_index, piece in enumerate(row):
                    row[piece_index] = self.TOKENS[piece]

        boardMatrix = np.array(boardMatrix, dtype=int)

        print(boardMatrix) if debug else None
        return boardMatrix

    def tokenizeSingle(self, piece: str):
        return self.TOKENS[piece]

    def tokenizeSingleSquare(self, square: str):
        square = square.strip().lower()

        if str(square[0]) not in self.FILES or square[1] not in self.RANKS:
            raise ValueError(
                square + " is not a valid square ",
                str(square[0]) not in self.FILES,
                int(square[1]) not in self.RANKS,
            )

        file = self.FILES.index(square[0])
        rank = self.RANKS.index(square[1])

        square = str(file) + str(rank)

        return int(square)

    def legalMoves(self):
        legal_moves = []
        for move in self.board.legal_moves:
            from_square = chess.square_name(move.from_square)
            to_square = chess.square_name(move.to_square)
            piece_moving = self.board.piece_at(move.from_square)
            legal_moves.append(
                (self.tokenizeSingle(str(piece_moving)), from_square, to_square)
            )

        return legal_moves


def square_name_to_index(square_name: str) -> int:
    """Convert a square name (e.g. 'e4') to an index 0-63."""
    file = ord(square_name[0]) - ord("a")
    rank = int(square_name[1]) - 1
    return rank * 8 + file


def move_to_index(move: Tuple[int, str, str]) -> int:
    """Convert a move tuple (piece_type, from_square, to_square) to a single index."""
    from_idx = square_name_to_index(move[1])
    to_idx = square_name_to_index(move[2])
    return from_idx * 64 + to_idx


def process_move(piece_type, from_square, to_square):
    return int(str(piece_type) + str(from_square) + str(to_square))


def main() -> None:
    chessHelperObj = ChessHelper()
    print(chessHelperObj.tokenizeSingleSquare("g3"))


if __name__ == "__main__":
    main()
