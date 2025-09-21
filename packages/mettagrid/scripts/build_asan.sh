#!/bin/bash

CFLAGS="-fsanitize=address -g" CXXFLAGS="-fsanitize=address -g" python setup.py build_ext --inplace --force
