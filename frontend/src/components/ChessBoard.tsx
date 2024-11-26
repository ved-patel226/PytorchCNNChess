import React, { useState, useCallback, useEffect } from "react";
import { Chessboard } from "react-chessboard";
import { Chess } from "chess.js";
import getAPI from "../functions/getAPI";

const Board: React.FC = () => {
  const [chess] = useState(new Chess());
  const [boardPosition, setBoardPosition] = useState(chess.fen());
  const [isErrorMove, setIsErrorMove] = useState(false);
  const [moveHistory, setMoveHistory] = useState<string[]>([]);
  const [evaluation, setEvaluation] = useState("0.0");

  const makeMove = useCallback(
    (sourceSquare: string, targetSquare: string) => {
      try {
        const move = chess.move({
          from: sourceSquare,
          to: targetSquare,
          promotion: "q",
        });

        if (move) {
          setBoardPosition(chess.fen());
          setIsErrorMove(false);
          setMoveHistory((prevHistory) => [
            ...prevHistory,
            move.from + move.to + (move.promotion || ""),
          ]);
          console.log(
            "Move in UCI format:",
            move.from + move.to + (move.promotion || "")
          );
          return true;
        }
        return false;
      } catch (error) {
        setIsErrorMove(true);
        return false;
      }
    },
    [chess]
  );

  const resetGame = () => {
    chess.reset();
    setBoardPosition(chess.fen());
    setEvaluation("0.0");

    getAPI({
      url: "api/reset",
    });
  };

  const isBlacksTurn = chess.turn() === "b";

  useEffect(() => {
    if (isBlacksTurn && moveHistory.length > 0) {
      const lastMove = moveHistory[moveHistory.length - 1];

      getAPI({
        url: `api/send/move/${lastMove}`,
      })
        .then((response) => {
          setBoardPosition(response);
          chess.load(response);
        })
        .catch((error) => {
          console.error(error);
        });

      getAPI({
        url: `api/get/move`,
      })
        .then((response) => {
          setBoardPosition(response[0]);
          chess.load(response[0]);

          setEvaluation(response[1]);
        })
        .catch((error) => {
          console.error(error);
        });
    }
  }, [isBlacksTurn, moveHistory, chess]);

  return (
    <div className="flex flex-col items-center p-4 space-y-4">
      <button className="btn btn-primary" onClick={resetGame}>
        Reset
      </button>
      <div className="w-full max-w-2xl">
        <Chessboard
          position={boardPosition}
          onPieceDrop={(sourceSquare, targetSquare) =>
            makeMove(sourceSquare, targetSquare)
          }
        />
      </div>

      <div className="text-4xl">Evaluation: {evaluation}</div>

      {isErrorMove && (
        <div className="text-red-500 text-3xl">Invalid move!</div>
      )}
      {chess.isCheckmate() && <div className="text-red-500">Checkmate!</div>}
      {chess.isDraw() && <div className="text-yellow-500">Draw!</div>}
      {isBlacksTurn && (
        <div className="">
          <span className="loading loading-spinner loading-lg"></span>
          <div className="text-blue-500 w-fit">AI THINKING</div>
        </div>
      )}
      <div className="move-history">
        <h3>Move History:</h3>
        <ul>
          {moveHistory.map((move, index) => (
            <li key={index}>{move}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default Board;
