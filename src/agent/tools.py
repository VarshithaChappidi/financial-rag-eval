import ast
import operator as op

from rag.generate import Reranker, detect_ticker
from rag.retrieve import Retriever

_retriever = None
_reranker = None


def _load():
    global _retriever, _reranker
    if _retriever is None:
        _retriever, _reranker = Retriever(), Reranker()
    return _retriever, _reranker


def search_filings(query: str, ticker: str | None = None, k: int = 5) -> list[dict]:
    """Hybrid retrieval + rerank over the bank 10-Ks. Returns chunks with their ids."""
    r, rr = _load()
    ticker = ticker or detect_ticker(query)
    candidates = r.hybrid(query, k=20, pool=40, ticker=ticker)
    return rr.rerank(query, candidates, k=k)


_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos,
}


def calculate(expression: str) -> float:
    """Safe arithmetic only: numbers, + - * / ** and parentheses."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"Unsupported expression: {expression}")
    return ev(ast.parse(expression, mode="eval").body)


if __name__ == "__main__":
    print(calculate("(16.9 - 13.4) / 13.4 * 100"))
    for c in search_filings("Citi net income 2025"):
        print(c["id"], c["ticker"], "|", c["section"])