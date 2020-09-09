#!/bin/bash

# Change to the folder with all images
directory=$1

for path in $directory; do
  path_1="$path/*"
  echo "$path_1"
  for path_2 in $path_1; do
    # For full conversion do each: image/svg, text/plain text/xml text/html
    if [[ `file --mime-type $path_2 | grep image/svg` ]]; then
        inkscape -z -e $path_2 $path_2
    fi

  done

done

for path in $directory; do
  path_1="$path/*"
  echo "$path_1"
  for path_2 in $path_1; do
    # For full conversion do each: image/svg, text/plain text/xml text/html
    if [[ `file --mime-type $path_2 | grep text/plain` ]]; then
        inkscape -z -e $path_2 $path_2
    fi

  done

done

for path in $directory; do
  path_1="$path/*"
  echo "$path_1"
  for path_2 in $path_1; do
    # For full conversion do each: image/svg, text/plain text/xml text/html
    if [[ `file --mime-type $path_2 | grep text/xml` ]]; then
        inkscape -z -e $path_2 $path_2
    fi

  done

done

for path in $directory; do
  path_1="$path/*"
  echo "$path_1"
  for path_2 in $path_1; do
    # For full conversion do each: image/svg, text/plain text/xml text/html
    if [[ `file --mime-type $path_2 | grep text/html` ]]; then
        inkscape -z -e $path_2 $path_2
    fi

  done

done
