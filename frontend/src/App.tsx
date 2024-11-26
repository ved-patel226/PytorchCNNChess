import getAPI from "./functions/getAPI";
import { useEffect } from "react";
import Board from "./components/ChessBoard";

function App() {
  useEffect(() => {
    getAPI({ url: "/api/get/board" })
      .then((response) => {
        console.log(response);
      })
      .catch((error) => {
        console.error(error);
      });
  }, []);

  return (
    <>
      <Board />
    </>
  );
}

export default App;
