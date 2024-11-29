import sys

sys.path.append("/mnt/Fedora2/code/python/tensorflow/rl/backend")

import chess
import torch

import warnings

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

import numpy as np
import chess.engine
from CHESS.essentials.func import ChessHelper, move_to_index
from CHESS.model.model import ChessPolicy
import chess


class CHESSAPI:
    def __init__(self, model_path):
        self.model = ChessPolicy().cuda()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.board = chess.Board()

        self.chess_helper = ChessHelper()

    def board_to_tensor(self, board):
        self.chess_helper.board = board

        board_tensor = self.chess_helper.tokenize()

        return torch.from_numpy(board_tensor).float().unsqueeze(0).cuda()

    def get_best_move(self):
        board_copy = self.board.copy()

        legal_moves = list(self.board.legal_moves)
        best_move = None
        best_score = -np.inf

        for move in legal_moves:
            board_copy.push(move)
            board_tensor = self.board_to_tensor(board_copy)
            board_copy.pop()

            with torch.no_grad():
                move_score = self.model(board_tensor).cpu().numpy().squeeze()

                if move_score > best_score:
                    best_score = move_score
                    best_move = move

        self.board.push(best_move)
        return best_score


def main() -> None:
    chess_ai = CHESSAPI("backend/chess_policy_modelV3.pth")

    print(chess_ai.get_best_move())


if __name__ == "__main__":
    main()
