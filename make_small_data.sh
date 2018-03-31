#!/usr/bin/env zsh

for i in `ls -d data/*/**`; do
  mkdir -p `echo $i | sed 's/data/data_small/'`
done

for i in `ls **/*.jpg`; do
  echo $i
  convert -geometry "299x299^" $i `echo $i | sed 's/data/data_small/'`
done

#