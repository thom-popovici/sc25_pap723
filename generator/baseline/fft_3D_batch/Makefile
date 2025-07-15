CXX=CC

HEFFTE_INCLUDE=../libraries/heffte/include/
HEFFTE_TESTS=../libraries/heffte/test/

HEFFTE_LIBS=../libraries/heffte/lib/

compile:
	${CXX} -O3 -std=c++11 -I${HEFFTE_INCLUDE} -fPIC -I${HEFFTE_TESTS} -L${HEFFTE_LIBS} fft3d_batch.cpp -o fft3d_batch.x -lheffte  -lcufft -lcudart -lfftw3 -lfftw3f -lfftw3_threads -lfftw3f_threads -pthread

clean:
	rm -rf *.x
