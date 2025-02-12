#!/bin/bash

LD_PRELOAD="$(gcc -print-file-name=libasan.so) $(gcc -print-file-name=libstdc++.so)" ASAN_OPTIONS=detect_leaks=1 python ./tests/test_leaks.py
