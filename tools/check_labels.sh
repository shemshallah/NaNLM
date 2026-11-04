#!/bin/sh
# Fail if a generated .nas file defines the same label twice.
#
# A duplicate label silently lets the last definition win. An earlier context
# runtime sent a hash loop into the hamming loop this way. This gate makes the
# failure loud before assembly.
set -eu
if [ $# -ne 1 ]; then
    echo "usage: $0 FILE.nas" >&2
    exit 2
fi
file=$1
if [ ! -f "$file" ]; then
    echo "check_labels: missing $file" >&2
    exit 1
fi
dups=$(awk '
    /^[A-Za-z_][A-Za-z0-9_]*:$/ {
        if (seen[$0]++) print $0
    }
' "$file")
if [ -n "$dups" ]; then
    echo "check_labels: duplicate labels in $file:" >&2
    printf '%s\n' "$dups" >&2
    exit 1
fi
exit 0
