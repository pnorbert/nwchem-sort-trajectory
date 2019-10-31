# nwchem-sort-trajectory

Standalone trajectory sorting code for the ADIOS2 output from NWChem. 


## How to build

Make sure MPI and ADIOS2 are installed
Assuming ADIOS2 is installed in /opt/adios2

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/opt/adios2 -DCMAKE_BUILD_TYPE=Release ..
$ make
$ cd ..
```


## How to run
This tool opens `<CASENAME>_trj_dump.bp` and writes `<CASENAME>_trj.bp`

```
$ mpirun -n 4 build/nwchem-sort-trajectory <CASENAME>

...


$ bpls -l <CASENAME>_trj.bp

```

