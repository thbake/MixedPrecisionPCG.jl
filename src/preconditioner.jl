using MATLAB

# Exported structs
export AbstractPreconditioner, FactorizationPreconditioner, PreconditioningScheme,
       Left, Right, Split

# Exported functions
export precondition

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
        Pr::AbstractMatrix) where {

            uS                    <: AbstractFloat,
            PreconditioningScheme <: Union{Left, Right}

        }

        new{uS, uS, PreconditioningScheme}(uS.(Pl), uS.(Pr))

    end

end

precondition(
    M::FactorizationPreconditioner{uL, uR, Left},
    v::Vector{u}) where {u, uL, uR } =  u.(M.Pr \ (M.Pl \ uL.(v)))

function precondition(
    M::FactorizationPreconditioner{uL, uR, Split},
    r::Vector{u}) where {u, uL, uR}

    println("I am being called")
    tmp =  M.Pl \ uL.(r)
    return M.Pr \ uR.(tmp)

end

precondition(Ms::AbstractMatrix{uS}, v::Vector{u}) where {uS <: AbstractFloat, u <:AbstractFloat} = u.(Ms \ uS.(v)) 

n = 10


A = sprand(n, n, 0.05)
M = A'A + 2I           # Get some sparse SPD matrix

# Construct preconditioner with incomplete Cholesky factorization.
L1 = mxcall(:ichol, 1, M)  # More expensive due to Julia/MATLAB intercommunication.
L2 = mat"ichol($M, struct('type', 'ict', 'droptol', 1e-2))"

# Later I could try implementing own incomplete Cholesky factorizations.
