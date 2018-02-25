#!/bin/bash
script='asteroidea_automated_learning.py'
dsetname='alarm'
for entry in datasets/*
do
    if [[ -d $entry ]]; then # check if $entry is a directory
        if [[ $entry == *"$dsetname"* ]]; then
            script_name="./tools/$script"
            echo "Running $script_name for dataset $entry..."
            python $script_name $entry
        fi
    fi
done
