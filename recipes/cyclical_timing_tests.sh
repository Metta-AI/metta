#!/bin/bash

# Cyclical Timing Study - Test different cyclical converter frequencies
# This script provides commands for testing individual timing configurations and combined testing

set -e

# Function to run individual timing test
run_individual_test() {
    local timing=$1
    echo "Running cyclical timing test for ${timing}-tick frequency..."
    ./devops/skypilot/launch.py train \
        run=$USER.cyclical_timing_${timing}.$(date +%m-%d) \
        trainer.curriculum=env/mettagrid/curriculum/cyclical_timing_${timing} \
        --gpus=1 \
        eval=null \
        "$@"
}

# Function to run all tests together
run_combined_test() {
    echo "Running combined cyclical timing study..."
    ./devops/skypilot/launch.py train \
        run=$USER.cyclical_timing_study.$(date +%m-%d) \
        trainer.curriculum=env/mettagrid/curriculum/cyclical_converter_timing_study \
        --gpus=1 \
        eval=null \
        "$@"
}

# Main execution
case "${1:-help}" in
    "6")
        run_individual_test 6 "${@:2}"
        ;;
    "8")
        run_individual_test 8 "${@:2}"
        ;;
    "10")
        run_individual_test 10 "${@:2}"
        ;;
    "16")
        run_individual_test 16 "${@:2}"
        ;;
    "20")
        run_individual_test 20 "${@:2}"
        ;;
    "50")
        run_individual_test 50 "${@:2}"
        ;;
    "100")
        run_individual_test 100 "${@:2}"
        ;;
    "all")
        run_combined_test "${@:2}"
        ;;
    "individual")
        echo "Running all individual timing tests..."
        run_individual_test 6 "${@:2}"
        run_individual_test 8 "${@:2}"
        run_individual_test 10 "${@:2}"
        run_individual_test 16 "${@:2}"
        run_individual_test 20 "${@:2}"
        run_individual_test 50 "${@:2}"
        run_individual_test 100 "${@:2}"
        ;;
    "help"|*)
        echo "Cyclical Timing Study Commands:"
        echo ""
        echo "Individual timing tests:"
        echo "  ./recipes/cyclical_timing_tests.sh 6    # Test 6-tick timing"
        echo "  ./recipes/cyclical_timing_tests.sh 8    # Test 8-tick timing"
        echo "  ./recipes/cyclical_timing_tests.sh 10   # Test 10-tick timing"
        echo "  ./recipes/cyclical_timing_tests.sh 16   # Test 16-tick timing"
        echo "  ./recipes/cyclical_timing_tests.sh 20   # Test 20-tick timing"
        echo "  ./recipes/cyclical_timing_tests.sh 50   # Test 50-tick timing"
        echo "  ./recipes/cyclical_timing_tests.sh 100  # Test 100-tick timing"
        echo ""
        echo "Combined testing:"
        echo "  ./recipes/cyclical_timing_tests.sh all        # Run all timing tests in one curriculum"
        echo "  ./recipes/cyclical_timing_tests.sh individual # Run all individual tests sequentially"
        echo ""
        echo "Manual commands (alternative approach):"
        echo ""
        echo "# Individual tests:"
        echo "./devops/skypilot/launch.py train run=\$USER.cyclical_timing_6.\$(date +%m-%d) trainer.curriculum=env/mettagrid/curriculum/cyclical_timing_6 --gpus=1 eval=null"
        echo "./devops/skypilot/launch.py train run=\$USER.cyclical_timing_8.\$(date +%m-%d) trainer.curriculum=env/mettagrid/curriculum/cyclical_timing_8 --gpus=1 eval=null"
        echo "./devops/skypilot/launch.py train run=\$USER.cyclical_timing_10.\$(date +%m-%d) trainer.curriculum=env/mettagrid/curriculum/cyclical_timing_10 --gpus=1 eval=null"
        echo "./devops/skypilot/launch.py train run=\$USER.cyclical_timing_16.\$(date +%m-%d) trainer.curriculum=env/mettagrid/curriculum/cyclical_timing_16 --gpus=1 eval=null"
        echo "./devops/skypilot/launch.py train run=\$USER.cyclical_timing_20.\$(date +%m-%d) trainer.curriculum=env/mettagrid/curriculum/cyclical_timing_20 --gpus=1 eval=null"
        echo "./devops/skypilot/launch.py train run=\$USER.cyclical_timing_50.\$(date +%m-%d) trainer.curriculum=env/mettagrid/curriculum/cyclical_timing_50 --gpus=1 eval=null"
        echo "./devops/skypilot/launch.py train run=\$USER.cyclical_timing_100.\$(date +%m-%d) trainer.curriculum=env/mettagrid/curriculum/cyclical_timing_100 --gpus=1 eval=null"
        echo ""
        echo "# Combined test:"
        echo "./devops/skypilot/launch.py train run=\$USER.cyclical_timing_study.\$(date +%m-%d) trainer.curriculum=env/mettagrid/curriculum/cyclical_converter_timing_study --gpus=1 eval=null"
        ;;
esac
