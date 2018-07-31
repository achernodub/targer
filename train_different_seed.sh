#!/bin/bash
for i in {1..5}
do
   python main.py -seed_num $i -report_fn report_seed_$i.txt
done
