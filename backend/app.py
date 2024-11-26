from flask import Flask, jsonify
from flask_cors import CORS
from CHESS import *
import chess

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

PATH = "/mnt/Fedora2/code/python/tensorflow/rl/backend/chess_policy_modelV2.pth"
chessAI = CHESSAPI(PATH)


@app.route("/api/get/board")
def API_GET_BOARD():
    return jsonify(chessAI.board.fen())


@app.route("/api/get/move")
def API_GET_MOVE():
    best_move_score = chessAI.get_best_move()

    return jsonify(chessAI.board.fen(), str(best_move_score))


@app.route("/api/reset")
def API_RESET():
    chessAI.board = chess.Board()
    return jsonify(chessAI.board.fen())


@app.route("/api/send/move/<move>")
def API_SEND_MOVE(move):
    if chess.Move.from_uci(move) in chessAI.board.legal_moves:
        chessAI.board.push(chess.Move.from_uci(move))
        return jsonify(chessAI.board.fen())
    else:
        chessAI.board = chess.Board()
        return jsonify(chessAI.board.fen())


def main() -> None:
    app.run(debug=True)


if __name__ == "__main__":
    main()
