#!/bin/bash
for i in {0..29}
do
    echo $i
    mv model$i.pth model.pth
    give cs9444 hw2 part1.py part2.py part3.py model.pth

    mv model.pth model$i.pth
done