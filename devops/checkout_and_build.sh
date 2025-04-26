#!/bin/bash

# This script checks out and builds all dependencies

if [ "$SKIP_BUILD" = "1" ]; then
    echo "SKIP_BUILD was set. Skipping checkout and build!"
    exit 0
fi

# Check if dependencies are already installed
if [ -n "$DEPS_INSTALLED" ]; then
    echo "Dependencies already installed. Skipping checkout and build!"
    exit 0
fi

# Exit immediately if a command exits with a non-zero status
set -e

# Repository URLs defined as variables
FAST_GAE_REPO="https://github.com/Metta-AI/fast_gae.git"
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
    
    # save any cached build files
    if [ -d "$repo_name" ]; then

        cd $repo_name
        echo "Repository content for $repo_name"
        ls -al
        cd ..

        echo "Moving existing repository to cache_$repo_name"
        mv "$repo_name" "cache_$repo_name"
    fi

    echo "Cloning $repo_name into $(pwd)"
    git clone $repo_url
    cd $repo_name

    echo "Fetching $repo_name into $(pwd)"
    git fetch

    echo "Checking out $branch branch for $repo_name"
    git checkout $branch

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

    echo "Repository content for $repo_name"
    ls -al
        
    # Check for build files
    echo "Checking for package files in $repo_name:"
    if [ -f "setup.py" ]; then
        echo "Found setup.py in root directory"
    elif [ -f "pyproject.toml" ]; then
        echo "Found pyproject.toml in root directory"
    else
        echo "No standard Python package files found in root directory"
    fi
    
    # Always run the build command regardless of structure
    echo "Building with command: $build_cmd"
    eval $build_cmd
    
    export PYTHONPATH="$PYTHONPATH:$(pwd)"
    echo "Updated PYTHONPATH: $PYTHONPATH"
    
    cd ..
    echo "Completed installation of $repo_name"
}


SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# ========== Main Project ==========
cd "$SCRIPT_DIR/.."

# Create and enter deps directory for all external dependencies
echo "Creating deps directory..."
mkdir -p deps
cd deps

# ========== METTAGRID ==========
# Note that version control for the mettagrid package has been brought into our monorepo
cd mettagrid
echo "Building mettagrid into $(pwd)"
python setup.py build_ext --inplace
pip install -e .
export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "Updated PYTHONPATH: $PYTHONPATH"
cd ..

# Install dependencies using the function
install_repo "fast_gae" $FAST_GAE_REPO "main" "python setup.py build_ext --inplace && pip install -e ."
install_repo "pufferlib" $PUFFERLIB_REPO "metta" "pip install ."
install_repo "carbs" $CARBS_REPO "main" "pip install -e ."
install_repo "wandb_carbs" $WANDB_CARBS_REPO "main" "pip install -e ."

# Mark dependencies as installed
export DEPS_INSTALLED=true
echo "Dependencies successfully installed and cached!"