using MixedPrecisionPCG
using Test, LinearAlgebra

@testset "MixedPrecisionPCG.jl" begin

    @testset "Compute error test" begin
        include("error_computation.jl")
    end
end
