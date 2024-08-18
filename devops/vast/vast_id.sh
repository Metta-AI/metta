#!/bin/bash

vastai show instances  | tail -n1 | awk '{print $1}'
