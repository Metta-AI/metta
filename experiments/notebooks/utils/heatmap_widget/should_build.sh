#!/usr/bin/env zsh

# Maintain the md5sums of the src/*.tsx files in a .md5sums file.
# If the md5sums have changed, rebuild the widget.


local md5sums_file=".md5sums"

# Get the first argument (should be 0 or 1) and set a variable to it
local build_and_save_md5sums="${1:-0}"

local current_md5sums=$(md5sum src/*.tsx)


# Build the project and save the md5sums of the src/*.tsx files.
function build_and_save() {
    local msg="$1"
    if [ "$build_and_save_md5sums" = "1" ]; then
        echo "$msg"
        echo "$current_md5sums" > "$md5sums_file"
        npx vite build
    fi
}


# If we don't have a compiled JS file at all, forget about checking md5sums and just build it.
if [ ! -f "heatmap_widget/static/index.js" ]; then
    build_and_save "No compiled JS file. Building and creating one..."
    exit 0

# If we have a md5sums file, check if the md5sums have changed, and, if so, build and save.
elif [ -f "$md5sums_file" ]; then
    local md5sums=$(cat "$md5sums_file")
    if [ "$md5sums" = "$current_md5sums" ]; then
        echo "src/*.tsx has not changed, skip build"
        exit 1
    else
        diff -d --color=always <(echo "$md5sums") <(echo "$current_md5sums")
        build_and_save "src/*.tsx has changed, rebuild"
        exit 0
    fi

# If we don't have a md5sums file, build the project and create one.
else
    build_and_save "No .md5sums file. Building and creating one..."
    exit 0
fi
