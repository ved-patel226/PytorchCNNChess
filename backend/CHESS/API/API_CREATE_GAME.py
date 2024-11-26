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

        board = self.board

        board_tensor = self.board_to_tensor(board)

        with torch.no_grad():
            policy = self.model(board_tensor)

        policy = policy.cpu().numpy().flatten()

        legal_moves = self.chess_helper.legalMoves()

        move_scores = {}
        for move in legal_moves:
            move_index = move_to_index(move)
            move_scores[move] = policy[move_index]

        best_move = max(move_scores, key=move_scores.get)
        best_move_score = move_scores[best_move]

        piece_type, from_square, to_square = best_move

        self.board.push(chess.Move.from_uci(f"{from_square}{to_square}"))
        return best_move_score


def main() -> None:
    chess_ai = ChessAI("backend/chess_policy_model.pth")
    print(chess_ai.get_best_move())
    print(chess_ai.get_best_move())
    chess_ai.board = chess.Board()
    print(chess_ai.get_best_move())


if __name__ == "__main__":
    main()
