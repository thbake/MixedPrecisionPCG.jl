module MixedPrecisionPCG

    using LinearAlgebra, SparseArrays
    using LinearAlgebra: dot, mul!

    include("preconditioner.jl")

    include("convergence.jl")

    include("pcg.jl")

end
