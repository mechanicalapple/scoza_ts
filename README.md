# scoza_ts
SCOZA TS - thermodynamic software based on self-consistent Ornstein-Zernike application

Compiling:

gcc -Wall -pedantic -std=gnu99 -Ofast -march=native -I/usr/local/include -L/usr/local/lib scoza_ts.c -lgsl -lgslcblas -lm -fopenmp -lpthread -lrt -lfftw3

Running:

./a.out ./log.txt ./in.txt