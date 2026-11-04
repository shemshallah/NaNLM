# SPDX-License-Identifier: MIT
"""Tiny deterministic text tokenizer for the fixture pipeline."""

from dataclasses import dataclass


@dataclass
class Token:
    kind: str
    text: str
    start: int
    end: int


def is_ident(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    while i < len(source):
        ch = source[i]
        if ch.isspace():
            start = i
            while i < len(source) and source[i].isspace():
                i += 1
            tokens.append(Token("space", source[start:i], start, i))
        elif ch.isalpha() or ch == "_":
            start = i
            while i < len(source) and is_ident(source[i]):
                i += 1
            tokens.append(Token("name", source[start:i], start, i))
        elif ch.isdigit():
            start = i
            while i < len(source) and source[i].isdigit():
                i += 1
            tokens.append(Token("number", source[start:i], start, i))
        else:
            i += 1
            tokens.append(Token("punct", ch, i - 1, i))
    return tokens


def render(tokens: list[Token]) -> str:
    return "".join(token.text for token in tokens)
