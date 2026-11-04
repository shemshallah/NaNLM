# code fixture

A small, original multi-language corpus used to test the continuous ingestion
pipeline. It is not training data for a production model. It exists so the
license gate, language tags, secret filter, minified-file filter, chunker, shard
builder, and runner-ring planner have deterministic inputs.

The parser below is deliberately simple and the code is deliberately varied:
C, Python, JavaScript, Rust, Go, shell, and SQL.
