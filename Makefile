IDIR=include
CC=g++
CFLAGS=--std=c++11 -I$(IDIR)
LIBS=-lm -larmadillo

render: src/* include/*
	$(CC) -o $@ src/* $(CFLAGS) $(LIBS)
