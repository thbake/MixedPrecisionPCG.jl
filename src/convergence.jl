export ConvergenceData
export  preconditioned_condition_number

mutable struct ConvergenceData{T}

    n                ::Int          # Number of dimension.
    max_iter         ::Int          # Number of iterations.
    updated_residuals::Matrix{T}   
    iterates         ::Matrix{T}    # n by k matrix representing the k iterates.

    function ConvergenceData{T}(n::Int, max_iter::Int) where T<:AbstractFloat

        new( n, max_iter, 
        ones(n, max_iter),
        ones(n, max_iter))

    end

end



function preconditioned_condition_number(
    A::AbstractMatrix,
    M::FactorizationPreconditioner{uL, uR, T}) where {uL, uR, T<:PreconditioningScheme}

    Aprec = precondition(M, A)

    return cond(Aprec)

end

