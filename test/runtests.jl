using MixedPrecisionPCG
using Test

@testset "MixedPrecisionPCG.jl" begin

    @testset "Compute error test" begin
        include("error_computation.jl")
    end
end
