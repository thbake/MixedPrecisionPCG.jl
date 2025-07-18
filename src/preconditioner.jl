using MATLAB

# Exported structs
export AbstractPreconditioner, AbstractSplit, FactorizationPreconditioner, 
       PreconditioningScheme, Left, Split, Right, SaadSplit

# Exported functions
export precondition, precondition!, getprecisions

"""
Construction and application of preconditioner.
"""

"""
Abstract tupe representing how the preconditioner will be applied.
"""
abstract type PreconditioningScheme end
abstract type AbstractSplit <: PreconditioningScheme end
struct Left      <: PreconditioningScheme end 
struct Right     <: PreconditioningScheme end 
struct Split     <: AbstractSplit end 
struct SaadSplit <: AbstractSplit end

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

    function FactorizationPreconditioner{uL, uR, scheme}(
        Pl    ::AbstractMatrix, 
        Pr    ::AbstractMatrix) where {uL <: AbstractFloat, uR <: AbstractFloat, scheme }

        new(uL.(Pl), uR.(Pr))

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

function getprecisions(preconditioner::FactorizationPreconditioner{uL, uR, AbstractSplit}) where {uL, uR}

    return eltype(preconditioner.Pl), eltype(preconditioner.Pr)

end

function getprecisions(preconditioner::FactorizationPreconditioner{uL, uR, Left}) where {uL, uR}

    return eltype(preconditioner.Pl)

end


precondition(
    M::FactorizationPreconditioner{uL, uR, SaadSplit},
    r::AbstractVector{u}) where {u, uL, uR} = M.Pr \ uR.(M.Pl \ uL.(r))

function precondition(Ms::AbstractMatrix{uS}, v::Vector{u}) where {uS <: AbstractFloat, u <:AbstractFloat} 

    return u.(Ms \ uS.(v)) 

end

function precondition!(
    M::FactorizationPreconditioner{uL, uR, SaadSplit},
    v::Vector{u}) where {u, uL, uR}

    v .= u.(M.Pl \ uL.(v)) # Change variable in place.
    p  = u.(M.Pr \ uR.(v))

    return p

end

function precondition(
    M::FactorizationPreconditioner{uL, uR, SaadSplit},
    A::AbstractMatrix) where{uL, uR}

    return inv(M.Pl) * A * inv(M.Pr) 

end

function general_precond(
    M::FactorizationPreconditioner{uL, uR, Left},
    r::AbstractVector{u}) where {u, uL, uR}

    s = u.(M.Pr \ (M.Pl \ uL.(r)))
    q = s
    z = r

    return s, q, z
end

function general_precond(
    M::FactorizationPreconditioner{uL, uR, Right},
    r::AbstractVector{u}) where {u, uL, uR}

    s = r
    q = u.(M.Pr \ M.Pl \ uR.(r))
    z = q

    return s, q, z
end

function general_precond(
    M::FactorizationPreconditioner{uL, uR, Split},
    r::AbstractVector{u}) where {u, uL, uR}

    s = u.(M.Pl \ uL.(r))
    q = u.(M.Pr \ uR.(s))
    z = s

    return s, q, z

end
