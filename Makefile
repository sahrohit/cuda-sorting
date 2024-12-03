CXX = nvcc
TARGETS = thrust singlethread multithread

.PHONY: all clean

all: $(TARGETS)

thrust: thrust.cu
	$(CXX) -o $@ $<

singlethread: singlethread.cu
	$(CXX) -o $@ $<

multithread: multithread.cu
	$(CXX) -o $@ $<

clean:
	rm -f $(TARGETS)
