-- SPDX-License-Identifier: MIT
CREATE TABLE IF NOT EXISTS shard (
    id          TEXT PRIMARY KEY,
    generation  INTEGER NOT NULL,
    runner      INTEGER NOT NULL,
    bytes       INTEGER NOT NULL,
    sha256      TEXT NOT NULL,
    state       TEXT NOT NULL CHECK (state IN ('leased', 'done', 'failed'))
);

CREATE INDEX IF NOT EXISTS shard_state ON shard (state, generation);

CREATE TABLE IF NOT EXISTS lease (
    shard_id    TEXT PRIMARY KEY REFERENCES shard (id),
    runner      INTEGER NOT NULL,
    expires_at  INTEGER NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 1
);

SELECT runner, count(*) AS assigned, sum(bytes) AS bytes
FROM shard
WHERE state = 'done'
GROUP BY runner
ORDER BY runner;
