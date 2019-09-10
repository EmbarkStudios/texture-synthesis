#!/usr/bin/env bash

# Run this script from your shell (git bash if you're on windows!)
# * pre-push - Fails to push if you haven't run `cargo fmt`

git config core.hooksPath .githooks
