import sys

sys.path.append("/mnt/Fedora2/code/python/tensorflow/rl")

import chess
import torch

import warnings

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

import numpy as np
import chess.engine
from CHESS.essentials.func import ChessHelper, move_to_index
from CHESS.model.model import ChessPolicy


class ChessAI:
    def __init__(self, model_path):
        self.model = ChessPolicy().cuda()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.chess_helper = ChessHelper()

    def board_to_tensor(self, board):
        self.chess_helper.board = board

        board_tensor = self.chess_helper.tokenize()

        return torch.from_numpy(board_tensor).float().unsqueeze(0).cuda()

    def get_best_move(self, board):
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

        piece_type, from_square, to_square = best_move
        return chess.Move.from_uci(f"{from_square}{to_square}")


def play_game():
    ai = ChessAI("chess_policy_model.pth")

    board = chess.Board(
        "2kr2nr/ppp2ppp/8/q2p4/1b1nP3/P1NB1P2/1P1B1P1P/R2QK2R w KQ - 3 12"
    )

    human_color = chess.WHITE
    ai_color = chess.BLACK

    print("Chess AI Game")
    print("Enter moves in standard algebraic notation (e.g., e2e4)")

    while not board.is_game_over():
        if board.turn == human_color:
            print("\nCurrent board:")
            print(board)

            while True:
                try:
                    move_str = input("Your move: ")
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move. Try again.")
                except ValueError:
                    print("Invalid move format. Use UCI notation.")

        else:
            print("\nAI is thinking...")
            ai_move = ai.get_best_move(board)
            board.push(ai_move)
            print(f"AI move: {ai_move}")

    print("\nGame Over!")
    print(board)
    print("Result:", board.result())


if __name__ == "__main__":
    play_game()
