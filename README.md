# Patch Match Stereo

Implementations in C++ of the Patch Match Stereo algorithm presented in:
* [PatchMatch Stereo - Stereo Matching with Slanted Support Windows](https://www.microsoft.com/en-us/research/wp-content/uploads/2011/01/PatchMatchStereo_BMVC2011_6MB.pdf)

Be aware, as a university project the code was not meant to be optimized but instead easy to read and understand.

Code was compiled and tested on ubuntu 16.04 usng g++ 5.4. Compilation requires the opencv libraries to be installed.
Compile it using make

Execute it using
```
pm path/to/imag1 path/to/image2
```