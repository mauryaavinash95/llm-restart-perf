#!/bin/bash 


TEST_DIR="/grand/VeloC/mikailg/file_scalability"

#sizes_M=(64 128 256 512)
sizesG=(1 2 4)

# for m in ${sizes_M[@]}; do
#     rm -rf $TEST_DIR
#     mkdir -p $TEST_DIR
#     lfs setstripe $TEST_DIR -S ${m}M
#     mpirun -np 4 ./write-only-thru "size${m}M-c{-1}"
# done

for g in ${sizes_G[@]}; do
    rm -rf $TEST_DIR
    mkdir -p $TEST_DIR
    lfs setstripe $TEST_DIR -S ${g}G
    mpirun -np 4 ./write-only-thru "size${g}G-c{-1}"
done


#CC -O2 -Wall -std=c++17 -I$LIBURING_PATH/include -I/home/mgossman/restart_perf/datastates-llm/include/common scalability_test.cpp -o write-only-thru -L$LIBURING_PATH/lib -luring -fopenmp -lcuda