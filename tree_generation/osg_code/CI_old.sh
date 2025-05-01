!/bin/bash

i=$1
i=$((i+1))
x=$(sed "${i}q;d" parameters.in | awk '{print $1}')
y=$(sed "${i}q;d" parameters.in | awk '{print $2}')
z=$(sed "${i}q;d" parameters.in | awk '{print $3}')
w=$(sed "${i}q;d" parameters.in | awk '{print $4}')

mkdir -p results/3_regular_graph
mkdir -p results/4_regular_graph
mkdir -p results/6_regular_graph
mkdir -p results/10_regular_graph

./clonal $x $y $z $w 100


