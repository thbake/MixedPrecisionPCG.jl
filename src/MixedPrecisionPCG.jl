module MixedPrecisionPCG

    using LinearAlgebra, SparseArrays
    using LinearAlgebra: dot, mul!

    include("preconditioner.jl")

    include("convergence.jl")

    include("system.jl")

    include("pcg.jl")

    include("data_computation.jl")



end
