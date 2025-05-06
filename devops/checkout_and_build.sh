#!/bin/bash

# This script checks out and builds all dependencies

if [ "$SKIP_BUILD" = "1" ]; then
    echo "SKIP_BUILD was set. Skipping checkout and build!"
    exit 0
fi

#
# Check if dependencies are already installed based on the presence of
# deps/.built (an empty file). If this file is found we will exit early.
#
# If we build the deps we will `touch "deps/.built"` at the end of this
# script. The `devops/setup_build` script removes this file so that the
# dependencies are reinstalled.
#
if [ -f "deps/.built" ]; then
    echo "Dependencies already installed. Skipping checkout and build!"
    echo "You can force reinstall by running \"devops/setup_build\""
    exit 0
fi

# Exit immediately if a command exits with a non-zero status
set -e

# Repository URLs defined as variables
PUFFERLIB_REPO="https://github.com/Metta-AI/pufferlib.git"
CARBS_REPO="https://github.com/kywch/carbs.git"
WANDB_CARBS_REPO="https://github.com/Metta-AI/wandb_carbs.git"

# Function to install a repository dependency
install_repo() {
    local repo_name=$1
    local repo_url=$2
    local branch=$3
    local build_cmd=$4

    echo "========== Installing $repo_name =========="

    if [ -d "$repo_name" ] && [ -d "$repo_name/.git" ]; then
        echo "Repository $repo_name already exists, updating instead of cloning"
        cd $repo_name
        echo "Current branch: $(git branch --show-current)"
        echo "Fetching updates for $repo_name"
        git fetch
        echo "Checking out $branch branch for $repo_name"
        git checkout $branch
        echo "Pulling latest changes"
        git pull origin $branch
    else
        # Repository doesn't exist or isn't a git repo, clone it
        if [ -d "$repo_name" ]; then
            echo "Directory $repo_name exists but is not a git repository"
            echo "Moving existing directory to cache_$repo_name"
            mv "$repo_name" "cache_$repo_name"
        fi

        echo "Cloning $repo_name from $repo_url"
        git clone $repo_url
        cd $repo_name
        echo "Checking out $branch branch for $repo_name"
        git checkout $branch

        # Restore build artifacts if we stored them before cloning
        if [ -d "../cache_$repo_name" ]; then
            echo "Attempting to restore cached build files"
            # Find and copy all *.so files
            find "../cache_$repo_name" -name "*.so" -exec cp {} . \;
            # Copy the build directory if it exists
            if [ -d "../cache_$repo_name/build" ]; then
                echo "Restoring build directory"
                cp -r "../cache_$repo_name/build" .
            fi
            # If there's a nested directory with the same name, check for build artifacts there too
            if [ -d "../cache_$repo_name/$repo_name" ]; then
                echo "Restoring nested build artifacts"
                mkdir -p "$repo_name"
                find "../cache_$repo_name/$repo_name" -name "*.so" -exec cp {} "$repo_name/" \;
            fi
            echo "Cached build files restored"
            # Cleanup the cache directory
            rm -rf "../cache_$repo_name"
        fi
    fi

    echo "Repository content for $repo_name"
    ls -al

    # Check for package files
    echo "Checking for package files in $repo_name:"
    if [ -f "setup.py" ]; then
        echo "Found setup.py in root directory"
    elif [ -f "pyproject.toml" ]; then
        echo "Found pyproject.toml in root directory"
    else
        echo "No standard Python package files found in root directory"
    fi

    # Run the build command
    echo "Building with command: $build_cmd"
    eval $build_cmd

    cd ..
    echo "Completed installation of $repo_name"
}


SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# ========== Main Project ==========
cd "$SCRIPT_DIR/.."

# ========== METTAGRID ==========
# Call the dedicated build_mettagrid.sh script instead of building directly
echo "Building mettagrid using devops/build_mettagrid.sh"
devops/build_mettagrid.sh

# Create and enter deps directory for all external dependencies
echo "Creating deps directory..."
mkdir -p deps
cd deps

# Install dependencies using the function
install_repo "pufferlib" $PUFFERLIB_REPO "metta" "pip install ." # pufferlib has problems with editable installation
install_repo "carbs" $CARBS_REPO "main" "pip install -e ."
install_repo "wandb_carbs" $WANDB_CARBS_REPO "main" "pip install -e ."

# Mark dependencies as installed
cd ..
touch "deps/.built"
echo "Dependencies successfully installed and cached!"
