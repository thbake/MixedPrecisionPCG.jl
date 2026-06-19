# Exported structs
export AbstractPreconditioner, AbstractSplit, FactorizationPreconditioner, 
        Left, PreconditioningScheme, Right, SaadSplit, Split

# Exported functions
export general_precond, getprecisions, precondition, precondition!

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

function scale_preconditioner(M::AbstractMatrix, precision::DataType)      

  n = size(M, 1)

  return precision.(M), I(n), I(n)

end

function scale_preconditioner(M::AbstractMatrix, precision::Type{Float16}) 

  M, R, S = two_sided_diagonal_scaling(M, 0.8, 1e-4)

  return M, R, S

end


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
    R ::AbstractMatrix
    S ::AbstractMatrix

	"""
	Constructor given left and right preconditioners, and preconditioning scheme.
	Precisions are deduced from type of each preconditioner.
	"""
    function FactorizationPreconditioner(
        Pl    ::AbstractMatrix{uL}, 
        Pr    ::AbstractMatrix{uR},
        scheme::Type{PreconditioningScheme}) where {uL, uR} <: AbstractFloat

        n = size(Pl, 1)
        new{uL, uR, scheme}(Pl, Pr, I(n), I(n)) 

    end

	"""
	Constructor given left and right preconditioners, and preconditioning scheme.
	Precisions are given explicitly.
	"""
    function FactorizationPreconditioner{uL, uR, scheme}(
        Pl    ::AbstractMatrix, 
        Pr    ::AbstractMatrix) where {uL <: AbstractFloat, uR <: AbstractFloat, scheme }

        # Perform scaling if necessary
        Pl, R, S = scale_preconditioner(Pl, uL)
        Pr, R, S = scale_preconditioner(Pl, uR)

        new(Pl, Pr, R, S)

    end

    """
    Constructor for one sided factorization-based preconditioner, i.e., full-left or full-right
	preconditioner.
    """
    #function FactorizationPreconditioner{uS, PreconditioningScheme}(
    #    Pl::AbstractMatrix,
    #    Pr::AbstractMatrix) where {uS <: AbstractFloat, PreconditioningScheme}

    #    new{uS, uS, PreconditioningScheme}(uS.(Pl), uS.(Pr), I)

    #end

end

function getprecisions(preconditioner::FactorizationPreconditioner{uL, uR, AbstractSplit}) where {uL, uR}

    return eltype(preconditioner.Pl), eltype(preconditioner.Pr)

end

function getprecisions(preconditioner::FactorizationPreconditioner{uL, uR, Left}) where {uL, uR}

    return eltype(preconditioner.Pl)

end

function getprecisions(::Type{Left}, precisions::Type{<:AbstractFloat})

    return precisions, precisions

end

function getprecisions(::Type{<:AbstractSplit}, precisions::Tuple{Type, Type})

    return precisions[1], precisions[2]
end


# Preconditioning methods for Saad's split PCG algorithm.
# =======================================================
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
# ============================================================

"""
Full-left preconditioning. Here uL = uR.
"""
function general_precond(
    M::FactorizationPreconditioner{uL, uR, Left},
    r::AbstractVector{u}) where {u, uL, uR}

    s = u.(M.Pr \ (M.Pl \ uL.(r)))
    q = s

    return s,q
end

function general_precond(
    M::FactorizationPreconditioner{Float16, Float16, Left},
    r::AbstractVector{u}) where {u}

    println("Scaling you")
    println(M.R)

    s_half = M.Pr \ (M.Pl \ Float16.(M.R * r)) # Scale, cast and solve
    s = u.(M.S * s_half) # Scale back and cast to working precision
    q = s

    return s,q
end

function general_precond(
    M::FactorizationPreconditioner{uL, uR, Right},
    r::AbstractVector{u}) where {u, uL, uR}

    s = r
    q = u.(M.Pr \ M.Pl \ uR.(s))

    return s, q
end

function general_precond(
    M::FactorizationPreconditioner{uL, uR, Split},
    r::AbstractVector{u}) where {u, uL, uR}

    s = u.(M.Pl \ uL.(r))
    q = u.(M.Pr \ uR.(s))

    return s, q

end
