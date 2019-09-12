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

function sub_run_examples() {
    prefix=${1:-0}

    cargo build --release
    target/release/texture-synthesis --out out/"${prefix}"1.jpg generate imgs/1.jpg
    target/release/texture-synthesis --rand-init 10 --seed 211 --in-size 300 -o out/"${prefix}"2.png generate imgs/multiexample/1.jpg imgs/multiexample/2.jpg imgs/multiexample/3.jpg imgs/multiexample/4.jpg
    target/release/texture-synthesis -o out/"${prefix}"3.png generate --target-guide imgs/masks/2_target.jpg --guides imgs/masks/2_example.jpg -- imgs/2.jpg
    target/release/texture-synthesis -o out/"${prefix}"4.png transfer-style --style imgs/multiexample/4.jpg --guide imgs/tom.jpg
    target/release/texture-synthesis --in-size 400 --out-size 400 --inpaint imgs/masks/3_inpaint.jpg -o out/"${prefix}"5.png generate imgs/3.jpg
    target/release/texture-synthesis --inpaint imgs/masks/1_tile.jpg --in-size 400 --out-size 400 --tiling -o out/"${prefix}"6.bmp generate imgs/1.jpg
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
