module NumericalExperiments

    using MixedPrecisionPCG
    using LinearAlgebra, SparseArrays

    include("scaling.jl")

    include("utils.jl")

    include("experiment_functions.jl")

    include("plot_general.jl")

    include("factorization_preconditioner.jl")


end
