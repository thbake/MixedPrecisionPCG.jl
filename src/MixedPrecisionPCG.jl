module MixedPrecisionPCG

    using LinearAlgebra, SparseArrays
    using LinearAlgebra: dot, mul!

    include("convergence.jl")

    include("preconditioner.jl")

    include("pcg.jl")

end
