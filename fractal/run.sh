#!/usr/bin/env bash
./main.py --branch="1:0.5:90" --branch="1:0.5:-90" 8 output/output.svg && inkview output/output.svg

