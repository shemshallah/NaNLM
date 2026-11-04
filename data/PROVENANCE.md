# Corpus provenance

`alice_excerpt.txt` is the first 30,000 bytes of *Alice's Adventures in
Wonderland* by Lewis Carroll, Project Gutenberg eBook #11. `alice_full.txt` is
the full 151,191-byte text.

- Source: https://www.gutenberg.org/files/11/11-0.txt
- Status: public domain in the United States.
- This repository stores a short excerpt only, for a byte-level model smoke
  test. It is not a benchmark claim.

The Gutenberg header/footer are included in the excerpt; the trainer does not
strip them. That is deliberate: the first model should not depend on a special
corpus cleaner.
