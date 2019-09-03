#!/bin/bash
set -e

if [ -z "$CRATES_TOKEN" ]; then
    echo "crates.io token not set"
    exit 1
fi

# Copy the README.md into both packages, as cargo package won't respect
# files above the Cargo.toml root
cp README.md lib
cp README.md cli

cargo publish --manifest-path "lib/Cargo.toml" --token "${CRATES_TOKEN}" --allow-dirty

# HACK: Wait for a few seconds and then force index update via fetch so the
# next step doesn't fail. Maybe check out carg-publish-all
sleep 5s
cargo fetch

cargo publish --manifest-path "cli/Cargo.toml" --token "${CRATES_TOKEN}" --allow-dirty
