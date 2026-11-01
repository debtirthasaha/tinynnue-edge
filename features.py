"""Position -> sparse feature indices, plus the material-count baseline.

Feature layout (769 binary inputs):
    index = plane * 64 + square        for each piece on the board
        plane = 0..5  white P,N,B,R,Q,K
        plane = 6..11 black P,N,B,R,Q,K
        square = 0..63 (a1=0, h8=63, python-chess numbering)
    index = 768                        if white is to move

At most 32 pieces + 1 side-to-move bit are active, so a position is a list of
<= 33 indices. That bound (MAX_ACTIVE) is also what the overflow analysis uses.
"""
import chess

N_FEATURES = 769
STM_FEATURE = 768
MAX_ACTIVE = 33
PAD = N_FEATURES  # padding index for fixed-width index arrays (row of zeros)

PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


def feature_index(square: int, piece: chess.Piece) -> int:
    plane = (0 if piece.color == chess.WHITE else 6) + piece.piece_type - 1
    return plane * 64 + square


def active_features(board: chess.Board) -> list[int]:
    feats = [feature_index(sq, p) for sq, p in board.piece_map().items()]
    if board.turn == chess.WHITE:
        feats.append(STM_FEATURE)
    return sorted(feats)


def feature_delta(before: list[int], after: list[int]) -> tuple[list[int], list[int]]:
    """Features to subtract and to add when going from `before` to `after`.

    Diffing the two sets covers every move type (castling moves two pieces,
    en passant removes a pawn from a third square, promotion swaps the piece
    plane, undo is just the reverse diff) without encoding any chess rules here.
    """
    b, a = set(before), set(after)
    return sorted(b - a), sorted(a - b)


def material_eval(board: chess.Board) -> int:
    """White-POV centipawns from piece counts only."""
    score = 0
    for p in board.piece_map().values():
        v = PIECE_VALUES[p.piece_type]
        score += v if p.color == chess.WHITE else -v
    return score
