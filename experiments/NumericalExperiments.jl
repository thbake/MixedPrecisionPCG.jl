module NumericalExperiments

    using MixedPrecisionPCG
    using LinearAlgebra

    include("eigenvalue_distributions.jl")

    include("plot_general.jl")

    include("factorization_preconditioner.jl")


end
