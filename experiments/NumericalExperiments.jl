module NumericalExperiments

    using MixedPrecisionPCG
    using LinearAlgebra, SparseArrays, Revise

    include("scaling.jl")

    include("utils.jl")

    include("experiment_functions.jl")

    include("plot_general.jl")

end
