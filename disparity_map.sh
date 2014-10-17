# clang++ -std=c++11 disparity_map.cpp `pkg-config --cflags --libs opencv`
rm ./a.out
g++ disparity_map.cpp -O3 `pkg-config --cflags --libs opencv`
# g++ disparity_map.cpp `pkg-config --cflags --libs opencv`
