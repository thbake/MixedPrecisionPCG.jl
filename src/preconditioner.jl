using MATLAB

# Exported structs
export AbstractPreconditioner, FactorizationPreconditioner, PreconditioningScheme,
       Left, Right, Split

# Exported functions
export precondition, precondition!, getprecisions

"""
Construction and application of preconditioner.
"""

"""
Abstract tupe representing how the preconditioner will be applied.
"""
abstract type PreconditioningScheme end
struct Left  <: PreconditioningScheme end 
struct Right <: PreconditioningScheme end 
struct Split <: PreconditioningScheme end 

"""
Abstract type representing a general notion of a preconditioner.
"""
abstract type AbstractPreconditioner{uL, uR, scheme} end

"""
Preconditioner resulting from an (incomplete) factorization of the system 
matrix A, where A = Pl ⋅ Pr, where Pl and Pr denote the left and right factors
in the decomposition.

For instance, such preconditioners could be the (incomplete) LU or Cholesky 
factorizations of A. That is, Pl = L and Pr = U, or Pl = L and Pr = L^T.

The parameter uL and uR denote the precision with which the preconditioner 
(factors) is (are) stored and applied, e.g., double (Float64), single 
(Float32), etc.

If the preconditioning scheme is left or right we have uL = uR.

"""
struct FactorizationPreconditioner{uL, uR, scheme} <: AbstractPreconditioner{uL, uR, scheme}

    Pl::Matrix{uL} # Left factor.
    Pr::Matrix{uR} # Right factor.

    function FactorizationPreconditioner(
        Pl    ::AbstractMatrix{uL}, 
        Pr    ::AbstractMatrix{uR},
        scheme::Type{PreconditioningScheme}) where {uL, uR} <: AbstractFloat

          new{uL, uR, scheme}(Pl, Pr) 

    end

    """
    Constructor for split factorization-based preconditioner.
    """
    function FactorizationPreconditioner{uL, uR, Split}(
        Pl::AbstractMatrix,
        Pr::AbstractMatrix) where { uL <: AbstractFloat, uR <: AbstractFloat }


        new( uL.(Pl), uR.(Pr) )

    end


    """
    Constructor for one sided factorization-based preconditioner.
    """
    function FactorizationPreconditioner{uS, PreconditioningScheme}(
        Pl::AbstractMatrix,
        Pr::AbstractMatrix) where {uS <: AbstractFloat, PreconditioningScheme}

        new{uS, uS, PreconditioningScheme}(uS.(Pl), uS.(Pr))

    end

end

function getprecisions(preconditioner::FactorizationPreconditioner{uL, uR, Split}) where {uL, uR}

    return eltype(preconditioner.Pl), eltype(preconditioner.Pr)

end

function getprecisions(preconditioner::FactorizationPreconditioner{uL, uR, Left}) where {uL, uR}

    return eltype(preconditioner.Pl)

end

precondition(
    M::FactorizationPreconditioner{uL, uR, Left},
    v::Vector{u}) where {u, uL, uR} =  u.(M.Pr \ (M.Pl \ uL.(v)))

precondition(
    M::FactorizationPreconditioner{uL, uR, Split},
    r::Vector{u}) where {u, uL, uR} = M.Pr \ uR.(M.Pl \ uL.(r))

function precondition(Ms::AbstractMatrix{uS}, v::Vector{u}) where {uS <: AbstractFloat, u <:AbstractFloat} 

    return u.(Ms \ uS.(v)) 

end

function precondition!(
    M::FactorizationPreconditioner{uL, uR, Split},
    v::Vector{u}) where {u, uL, uR}

    v .= u.(M.Pl \ uL.(v)) # Change variable in place.
    p  = u.(M.Pr \ uR.(v))

    return p

end
