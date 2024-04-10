#!/bin/bash

mkdir -p results vtk

python3 convergence_study.py -Lx 6 -Lt 7 -bdf 1 -ex 'travel_cir' -vtk 0
python3 convergence_study.py -Lx 6 -Lt 7 -bdf 2 -ex 'travel_cir' -vtk 0

python3 convergence_study.py -Lx 6 -Lt 7 -bdf 1 -ex 'kite' -vtk 0
python3 convergence_study.py -Lx 6 -Lt 7 -bdf 2 -ex 'kite' -vtk 0

python3 convergence_study.py -Lx 1 -Lt 1 -bdf 1 -ex 'collide' -vtk 1

python3 convergence_study.py -Lx 1 -Lt 1 -bdf 1 -ex 'collide3d' -vtk 1
