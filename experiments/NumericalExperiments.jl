module NumericalExperiments

    using MixedPrecisionPCG
    using LinearAlgebra, SparseArrays

    include("utils.jl")

    include("plot_general.jl")

    include("factorization_preconditioner.jl")


end
