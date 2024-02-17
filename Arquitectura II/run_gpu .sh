#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) Se está compilando la búsqueda de caminos más cortos entre pares en paralelo- 1 of 7 gpu_practice.cpp
icpx -fsycl lab/gpu_practice.cpp
if [ $? -eq 0 ]; then ./a.out; fi

