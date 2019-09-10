#!/bin/bash
set -e

name=$(basename "$0")
root=$(realpath "$(dirname "$(dirname "$0")")")
target_dir="$root/lib/target"

if ! [ -x "$(command -v critcmp)" ]; then
    echo "Installing critcmp..."
    cargo install critcmp
fi

function sub_help() {
    printf "Usage: %s <subcommand> [options]\n" "$name"
    echo "Subcommands:"
    echo "    export    Exports the specified baselines to json in the benches dir"
    echo "    cmp       Compare two benchmarks by name"
    echo ""
    echo "For help with each subcommand run:"
    echo "$name <subcommand> -h | --help"
    echo ""
}

function sub_cmp() {
    critcmp --target-dir "$target_dir" "$@"
}

function sub_export() {
    critcmp --target-dir "$target_dir" --export "$1" > "$root/lib/benches/${2:-$1}.json"
}

subcommand=$1
case $subcommand in
    "" | "-h" | "--help")
        sub_help
        ;;
    *)
        shift
        "sub_${subcommand}" "$@"
        if [ $? = 127 ]; then
            echo "Error: '$subcommand' is not a known subcommand." >&2
            echo "       Run '$name --help' for a list of known subcommands." >&2
            exit 1
        fi
        ;;
esac