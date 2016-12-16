CXX = g++
CXXFLAGS = -std=c++0x -Wall -O2 -I/usr/local/sllib/include -I/usr/local/gsl/include
LDFLAGS = -L/usr/local/sllib/lib64 -L/usr/local/gsl/lib
LDLIBS = -lgsl -lopenblas -lm -lsllib
TARGET = L1 L2

SRCS_L1 = main_L1.cc src/arg_option.cc src/read_file.cc src/util.cc src/L1.cc
OBJS_L1 = $(SRCS_L1:.cc=.o)
SRCS_L2 = main_L2.cc src/arg_option.cc src/read_file.cc src/util.cc src/L2.cc
OBJS_L2 = $(SRCS_L2:.cc=.o)

all: $(TARGET)

#$(TARGET): $(OBJS)
#	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)

L1: $(OBJS_L1)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)
L2: $(OBJS_L2)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)

.PHONY: clean
clean:
	rm -f L1 $(OBJS_L1)
	rm -f L2 $(OBJS_L2)
	rm -f *.dat