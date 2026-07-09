# Optimization

## Overview

Optimization is the process of making a system, design, or decision as effective as possible by minimizing or maximizing one or more objective functions subject to constraints.

## Key Concepts

- **Objective Function**: The function to be minimized or maximized
- **Constraints**: Conditions that must be satisfied by the solution
- **Feasible Region**: The set of all points satisfying all constraints
- **Local vs Global Optima**: Local optima are best within a neighborhood; global optima are best overall

## Types of Optimization

### 1. Continuous Optimization
- **Unconstrained**: No constraints on variables
- **Constrained**: Subject to equality and/or inequality constraints
- **Linear Programming (LP)**: Linear objective and linear constraints
- **Nonlinear Programming (NLP)**: Nonlinear objective or constraints

### 2. Discrete Optimization
- **Integer Programming**: Variables restricted to integer values
- **Combinatorial Optimization**: Finding optimal object from finite set
- **Graph Optimization**: Shortest path, maximum flow, minimum spanning tree

### 3. Convex vs Non-Convex
- **Convex**: Any local optimum is a global optimum
- **Non-Convex**: Multiple local optima; harder to solve

## Common Algorithms

| Algorithm | Type | Use Case |
|-----------|------|----------|
| Gradient Descent | Continuous, Unconstrained | Deep learning, regression |
| Newton's Method | Continuous, Unconstrained | Small-scale problems |
| Simplex Method | Linear Programming | Resource allocation |
| Interior Point Methods | Linear/Nonlinear | Large-scale LP |
| Genetic Algorithms | Metaheuristic | Complex, non-differentiable |
| Simulated Annealing | Metaheuristic | Combinatorial optimization |
| Particle Swarm Optimization | Metaheuristic | Continuous optimization |
| Branch and Bound | Discrete | Integer programming |

## Complexity

Optimization problems can vary a lot in complexity. Some, like linear programming, can often be solved efficiently with well-known algorithms, while others, like integer programming and many non-convex problems, can become very expensive as the input size grows.

- **Time complexity**: How the runtime grows with the number of variables, constraints, and dimensions
- **Space complexity**: How much memory is needed to store data, gradients, matrices, or search states
- **Problem structure**: Convex problems are usually easier to solve than non-convex ones
- **Scalability**: Large datasets and high-dimensional spaces can make even simple methods slow

In practice, the chosen algorithm depends on both solution quality and efficiency. A fast approximate method may be better than an exact method when the problem is large or too complex to solve optimally.

## Applications

- **Machine Learning**: Training models (loss minimization)
- **Operations Research**: Supply chain, scheduling, logistics
- **Engineering Design**: Structural optimization, control systems
- **Finance**: Portfolio optimization, risk management
- **Economics**: Utility maximization, cost minimization

## Software Tools

- **SciPy.optimize** (Python)
- **CVXPY** (Python) — convex optimization
- **Gurobi / CPLEX** — commercial solvers
- **OR-Tools** (Google) — constraint programming
- **Pyomo** — optimization modeling language

## Quick Example: Route Planning

Imagine a delivery app that needs the fastest route between two locations.
The app defines an objective function for travel time, adds constraints such as traffic and road closures, then searches for the path with the lowest cost.

This is the same optimization idea used in scheduling, resource allocation, and model training: define a goal, respect constraints, and search for the best solution.

## `nproc` (CPU cores and parallelism)

### Overview

`nproc` is a small Unix utility that reports the number of available processing units (CPU cores/threads) on the system. By default it returns the number of processing units available to the current process (honoring CPU affinity and container/cgroup limits); use `nproc --all` to see the total number of processors on the host regardless of affinity.

### Why it matters for optimization

- **Parallel speedup:** Many optimization routines (gradient computation, population-based metaheuristics, search/beam evaluation) can be parallelized. Knowing the number of available processors helps size worker pools and batch jobs.
- **Avoiding oversubscription:** Spawning more CPU-bound threads than available processing units causes context switching and reduces throughput. For CPU-bound workloads, aim for at most the number of physical cores (or logical processors if hyperthreading benefits the workload).
- **Container-aware behavior:** Inside containers or under cgroup limits, `nproc` reflects the constrained CPU quota (unless `--all` is used). This avoids accidentally launching too many workers inside orchestrated environments.

### Relevant concepts

- **Physical vs logical cores:** Modern CPUs expose logical processors (hyperthreads) — use `lscpu` to inspect `Core(s) per socket` vs `Thread(s) per core`.
- **Affinity and cgroups:** A process may be restricted to a subset of CPUs via `taskset` (affinity) or container runtime/cgroups; `nproc` by default observes those restrictions.
- **Amdahl's law:** Theoretical speedup when parallelizing a fraction $p$ of work across $n$ processors is

$$
S(n) = \frac{1}{(1-p) + p/n}
$$

This shows diminishing returns as $n$ increases unless $p$ is very close to 1.

### Practical recommendations

- For **CPU-bound** optimization tasks, start with `workers = $(nproc)` or `workers = $(nproc --ignore=1)` to leave one core free for system tasks. Prefer using the number of physical cores when hyperthreading gives little benefit.
- For **IO-bound** tasks, you can use more workers than `nproc` because threads often block on IO.
- When running in containers, prefer `nproc` (without `--all`) so your worker count adapts to the allocated quota.
- Expose worker count as a configuration value or environment variable (e.g., `WORKERS=$(nproc)`) and allow overriding for tuning.

### Pitfalls and advanced notes

- **Oversubscription:** Creating too many threads/processes can hurt latency-sensitive workloads.
- **NUMA:** On NUMA systems, spreading workers across sockets may increase memory access latency; consider socket-aware placement for large-memory workloads.
- **Benchmark before scaling:** Use microbenchmarks to confirm whether increasing threads or using hyperthreads helps your specific optimization algorithm.

## Website Notes

- Add a simple chart or diagram showing objective, constraints, and optimal solution.
- Include a short animation or step-by-step flow for visitors who prefer visual explanations.
- Pair this page with a real-world example so the concept feels practical right away.
