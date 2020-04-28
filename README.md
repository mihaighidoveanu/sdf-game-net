# SDF-Net

## Setup
Run *python port_ccode.py* to port C libraries to python. This will compile the C code as a binary file (.so on UNIX) that Python will further use.
See fontloader.py as an example of how to use the ksv\_truetype library.

To expose additional functions, their implementation should be in fontloader.h and their declaration in port\_ccode.py!

