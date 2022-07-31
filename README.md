# lgca
A Matlab/CUDA based simulation of infectious contacts using a Lattice Gas Cellular Automaton

An implementation based on Matlab and CUDA.
Three CUDA kernels are available and can be used interchangebly.
The key difference is how the chiral rotations at collisions of agents is handled.

* lgca_modulo_collissions.cu: rotation direction based on modulo division of the thread-id
* lgca_monotone_collissions.cu: monotone rotation (always counter-clockwise)
* lgca_no_collissions.cu: no rotations after colissions

For more information see:
A) Attached master thesis of myself
B) Dieter A. Wolf-Gladrow: Lattice Gas Cellular Automata and Lattice Boltzmann Models
ISBN: 978-3-540-66973-9
