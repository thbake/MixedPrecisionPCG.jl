# MixedPrecisionPCG

[![Build Status](https://github.com/thbake/MixedPrecisionPCG.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/thbake/MixedPrecisionPCG.jl/actions/workflows/CI.yml?query=branch%3Amain)

Code for reproducing results in T. Bake, E. Carson and Y. Ma (2025), Forward and backward error bounds for a mixed precision preconditioned conjugate gradient algorithm.

## Running experiments
```julia
julia> include("experiments/NumericalExperiments.jl")

julia> using .NumericalExperiments

julia> include("experiments/experiments.jl")
```

This will create a directory called "json\_data" with the corresponding .json files containing convergence data.
Plots were created with the Typst package Lilaq.
