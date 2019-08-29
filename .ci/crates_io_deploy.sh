#!/bin/bash
set -e

if [ -z "$CRATES_TOKEN" ]; then
    echo "crates.io token not set"
    exit 1
fi

cargo publish --manifest-path "lib/Cargo.toml" --token "${CRATES_TOKEN}"
cargo publish --manifest-path "cli/Cargo.toml" --token "${CRATES_TOKEN}"
