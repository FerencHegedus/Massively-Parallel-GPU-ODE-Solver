# Massively-Parallel-GPU-ODE-Solver
GPU accelerated integrator for a large number of independent ordinary differential equation systems

## Modules:

### Single System Per-Thread v3.1
It solves a large number of instances of the same ODE system with different initial conditions and/or parameter sets.

### Coupled Systems Per-Block v1.0
It solves a large number of instances of a coupled system (composed by many subsystems) with different initial conditions and/or parameter sets.

## Release notes:

### August 18, 2020.
Single System Per-Thread v3.1:
* Support for both single and double precision.
* Improved/reworked manual, e.g., a more detailed installation guide for Windows users.
* Inclusion of the batch file make.bat to simplify the compilation process on Windows.
* A new tutorial (Tutorial 5) is added to test an example having instances with very large time scale differences. The performance curve of MPGOS is compared with program packages odeint (C++) and DifferentialEquations.jl (Julia). MPGOS is superior over these packages.
* Tutorial 6 (impact dynamics) is also extended with performance curves, see the previous point. In this case, MPGOS is the only program package that is capable of handling systems with impact dynamics (on GPUs). Therefore, performance comparisons were made only with CPU versions of odeint and DifferentialEquations.jl.
* Minor changes: Clear separation of the TimeDomain and the ActualState variables.

Coupled Systems Per-Block v1.0:
* This first MPGOS module is ready. The code is designed to solve a large number of instances of a coupled system (composed by many subsystems called units) with different initial conditions and/or parameter sets.
* The module is inherited almost all the features of the module Single System Per-Thread v3.1. There are few specialities, e.g., event handling is possible only on a unit level; only explicit coupling can be treated in a matrix form; for details, the interested reader is referred to the manual.
* Two tutorial examples are provided.

### February 14, 2020.
#### Single System Per-Thread v3.0:
* Massive performance improvements.
* The introduced template metaprogramming technique allowed us to produce a highly optimised code.
* The average seep-up is 3x, while for low dimensional systems, it can be even an order of magnitude.

### October 10, 2019.
#### Single System Per-Thread v2.1:
* With Template Metaprogramming, the code is fully templatized to generate highly specialised solver code during compile time (as a function of the algorithm and the necessity of event handling and dense output). Accordingly, the file system is reworked.
* Small extension: possibility to use integers shared parameters and integer accessories to be able to achieve complex indexing techniques efficiently for complicated systems.

### August 13, 2019.
#### Single System Per-Thread v2.0:
* Dense output is now supported with few limitations, see the manual. This is a prerequisit e.g. for solving delay differential equations.
* The code and its interface is greatly simplified and cleared. For instance, the Problem Pool is completely omitted from the code (it was kept for historical reason), and many possible options are now bound to the Solver Object that can be setup all with a single member function.
* The manual is also restructured and simplified according to the feedbacks.

### April 9, 2019.
#### Single System Per-Thread v1.1:
* A device (GPU) can be associated to each Solver Object. Thus, device selection is now handled automatically.
* A CUDA stream is automatically created for each Solver Object.
* New set of member functions to overlap CPU-GPU computations, and to easily distribute workload to different GPUs in a single node. This includes asynchronous memory and kernel operations, and synchronisation possibilities between CPU threads and GPU streams.
* An active number of threads variable can be specified in each integration phase to handle the tailing effect comfortably.
* Two new tutorial examples are added: a) overlapping CPU and GPU computations using multiple Solver Objects b) using multiple GPUs available in a single machine/node.

### February 14, 2019.
#### Single System Per-Thread v1.0:
* This first MPGOS module is ready. The code is designed to solve a huge number of independent but identical (the parameter sets and the initial conditions can be different) ODE systems on GPUs.
* User-friendliness. Even those who are new to C++ programming, only a short course is more than enough to use the program package.
* There is a detailed manual with tutorial examples. Therefore, the user can easily build-up its own project by copy-paste code blocks.
* Efficient and robust event handling.
* User-defined action after every time step for flexibility.
* User-defined "interactions" after every successful time step or event handling (very useful e.g. for impact dynamics, see the tutorial examples in the manual).
* Possibility to utilize the GPU's memory hierarchy without explicit knowledge on the details.
* User-programmable parameter for flexible implementations and storing special property of a trajectory.
* Only explicit solvers: 4th order Runge-Kutta with a fixed time step, and 4th order Runge-Kutta-Cash-Karp method with 5th order embedded error estimation. (due to the complex control flow of implicit solvers, explicit solver sometimes performs better than the implicit ones even for stiff problems).
* Only double precision arithmetic operations are supported.
* Storing only the endpoints of each integration phase (in order to improve speed). However, this is rarely a problem, as the user-programmable parameters and the aforementioned user-defined interactions allow to store the most complex properties of a trajectory, see the documentation.
