IDIR=include
CC=g++
CFLAGS=--std=c++11 -I$(IDIR) $(FLAGS)
LIBS=-lm -larmadillo

render: src/* include/*
	$(CC) -o $@ src/* $(CFLAGS) $(LIBS)
