CC=g++
LIBS=-lgsl -lgslcblas -std=c++11



seir_simulator: seir_simulator.cpp seir_parameters.h
	@echo Compiling $<...
	$(CC) -o $@ $< $(LIBS)

seir_simulator_gamma: seir_simulator_gamma.cpp seir_parameters_gamma.h
	@echo Compiling $<...
	$(CC) -o $@ $< $(LIBS)

		
