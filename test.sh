#!/bin/bash

CURRENT_DIR=$(pwd)
SCRIPT_DIR="$CURRENT_DIR/benchmark"

run_test() {
	printf "\e[1;31m[NOTE] Enter JUST AN ENTER to quit FZF.\n\e[0m"
	cd "$SCRIPT_DIR" || exit
	while true; do
		dir=$(find . -type d | grep -v "__pycache__" | grep -v "mlpackage" | grep -v "assets" | fzf)
		if [ "$dir" == "." ]; then
			printf "[INFO] Exiting...\n"
			exit 0
		elif [ -d "$dir" ]; then
			printf "[INFO] Running unittest in \e[1m\e[36m%s\e[37m\e[0m\n" "$dir"
			printf "\n"
			python3 -m unittest discover -s "$dir"
			rm -rf imgui.ini
			break
		else
			printf "[ERROR] Directory not found.\n"
		fi
	done
	cd "$CURRENT_DIR" || exit

}

run_test
