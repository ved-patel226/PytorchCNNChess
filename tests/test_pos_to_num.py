from typing import Tuple


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


def test_conversion():

    assert square_name_to_index("a1") == 0
    assert square_name_to_index("h1") == 7
    assert square_name_to_index("a8") == 56
    assert square_name_to_index("h8") == 63
    assert square_name_to_index("e4") == 28

    move = (1, "e2", "e4")
    move_idx = move_to_index(move)
    print(f"Move {move} converted to index {move_idx}")


if __name__ == "__main__":
    test_conversion()
