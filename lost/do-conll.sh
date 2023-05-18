#!/bin/sh

./run.sh
etc/out2tab.lua < output.out > output.txt
paste dat/test.txt output.txt | cut -f 1,2,3,5 | tr '\t' ' ' | etc/conlleval.perl > output.evl

cat output.evl

