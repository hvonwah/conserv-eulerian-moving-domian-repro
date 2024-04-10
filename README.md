[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10951768.svg)](https://doi.org/10.5281/zenodo.10951768)

This repository contains the python scripts to reproduce the computational examples presented in "A conservative Eulerian finite element method for transport and diffusion in moving domains" by M. Olshanskii and H. v. Wahl.

# Files
```
|- README.md                                                    // This file
|- LICENSE                                                      // The licence file
|- install.txt                                                  // Installation help
|- convergence_study.py                                         // Convergence study driver
|- eulerian_moving_domain.py                                    // Main implementation of method
|- example_colliding.py                                         // Example parameters 
|- example_colliding3d.py                                       // Example parameters     
|- example_kite.py                                              // Example parameters
|- example_translation_circle.py                                // Example parameters
|- run_examples.bash                                            // Script to run all studies
```

# Installation
Detailed instructions and the specific version of `NGSolve` and `ngsxfem` used are given in `install.txt`.

# How to reproduce
The methods presented in our work are implemented in `eulerian_moving_domain.py`. The specific example parameters are set in `example_*.py` and the convergence loop is computed in `convergence_study.py` and the script `run_examples.bash` can be used to run the studies presented.
