.PHONY: build clean
build: clean setup.py matmulmodule.c
	CC=gcc python3 setup.py build

run: build client.py
	PYTHONPATH=build/lib.linux-x86_64-cpython-311/ python3 client.py	

test: build test_general.py
	ASAN_OPTIONS=detect_leaks=0 \
		LD_PRELOAD=`gcc -print-file-name=libasan.so` \
		PYTHONPATH=build/lib.linux-x86_64-cpython-311/ \
		pytest -s

clean:
	python3 setup.py clean --all
