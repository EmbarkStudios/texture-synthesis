#!/bin/bash
set -e

cargo publish --manifest-path "lib/Cargo.toml" --token "${1}"
cargo publish --manifest-path "cli/Cargo.toml" --token "${1}"
