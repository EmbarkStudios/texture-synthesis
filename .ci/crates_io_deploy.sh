#!/bin/bash
set -e

if [ -z "$CRATES_TOKEN" ]; then
    echo "crates.io token not set"
    exit 1
fi

cargo publish --manifest-path "lib/Cargo.toml" --token "${CRATES_TOKEN}"

# HACK: Wait for a few seconds and then force index update via fetch so the
# next step doesn't fail. Maybe check out carg-publish-all
sleep 5s
cargo fetch

cargo publish --manifest-path "cli/Cargo.toml" --token "${CRATES_TOKEN}"
