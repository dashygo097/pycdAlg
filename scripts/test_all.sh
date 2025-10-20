#!/bin/bash

BASE_DIR=$(dirname $(cd "$(dirname "$0")" && pwd))
BENCHMARK_DIR="$BASE_DIR/benchmark"

run_all_tests() {
    printf "\e[1;32m[INFO] Running ALL unit tests...\e[0m\n"
    cd "$BENCHMARK_DIR" || exit
    
    test_dirs=$(find . -type f -name "*test*.py" -exec dirname {} \; | grep -v "__pycache__" | grep -v "datasets" | sort | uniq)
    
    if [ -z "$test_dirs" ]; then
        printf "\e[1;33m[WARNING] No test directories found!\e[0m\n"
        cd "$BASE_DIR" || exit
        return 1
    fi
    
    printf "\e[1;36mFound test directories:\e[0m\n"
    echo "$test_dirs"
    printf "\n"
    
    total_dirs=0
    passed_dirs=0
    total_tests=$(echo "$test_dirs" | wc -l)
    
    while IFS= read -r dir; do
        if [ -n "$dir" ] && [ "$dir" != "." ]; then
            ((total_dirs++))
            printf "\e[1;34m[TEST %d/%d] Running unittest in: \e[1;35m%s\e[0m\n" "$total_dirs" "$total_tests" "$dir"
            printf "%.0s-" {1..50}
            printf "\n"
            
            if python3 -m unittest discover -s "$dir" -v; then
                printf "\e[1;32m✓ PASSED: %s\e[0m\n\n" "$dir"
                ((passed_dirs++))
            else
                printf "\e[1;31m✗ FAILED: %s\e[0m\n\n" "$dir"
            fi
            
            rm -rf imgui.ini 2>/dev/null || true
        fi
    done <<< "$test_dirs"

    printf "%.0s=" {1..60}
    printf "\n"
    printf "\e[1;36mTEST SUMMARY:\e[0m\n"
    printf "Total test directories: %d\n" "$total_dirs"
    printf "Passed: \e[1;32m%d\e[0m\n" "$passed_dirs"
    printf "Failed: \e[1;31m%d\e[0m\n" "$((total_dirs - passed_dirs))"
    
    if [ "$passed_dirs" -eq "$total_dirs" ]; then
        printf "\e[1;32m ALL TESTS PASSED!\e[0m\n"
    else
        printf "\e[1;33m SOME TESTS FAILED!\e[0m\n"
    fi
    
    cd "$BASE_DIR" || exit
    return $((total_dirs - passed_dirs))
}

run_all_tests
