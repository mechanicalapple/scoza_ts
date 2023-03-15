# scoza_ts
SCOZA TS - thermodynamic software based on self-consistent Ornstein-Zernike application.

Compiling:

    gcc -Wall -pedantic -std=gnu99 -Ofast -march=native -I/usr/local/include -L/usr/local/lib scoza_ts.c -lgsl -lgslcblas -lm -fopenmp -lpthread -lrt -lfftw3

Running:

    ./a.out ./log.txt ./in.txt

Citing:

[https://doi.org/10.1134/S1990793114010023](https://doi.org/10.1134/S1990793114010023)

[https://doi.org/10.1088/1742-6596/653/1/012055](https://doi.org/10.1088/1742-6596/653/1/012055)

[https://doi.org/10.1016/j.phpro.2015.09.102](https://doi.org/10.1016/j.phpro.2015.09.102)
