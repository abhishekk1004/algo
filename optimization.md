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