# Native Nsight Reports

Profile runs write only native NVIDIA report files to this directory:

- `*.nsys-rep`: Nsight Systems application timeline reports.
- `*.ncu-rep`: Nsight Compute kernel analysis reports.

The CI artifact includes only these two report formats. Nsight Compute profiling runs when a GPU is available and captures a short kernel sample.
