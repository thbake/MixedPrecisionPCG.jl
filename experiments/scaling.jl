using LinearAlgebra

export two_sided_diagonal_scaling

"""
Symmetry-preserving row and column equilibration.

Given a real n times n matrix A, compute nonsingular diagonal matrices R and S
such that B = RAS has the property that its largest entry is equal to one,
and R = S if A = Aᵀ.

"""
function row_col_equilibration(A, tol::AbstractFloat)

    n = size(A, 1)

    tmpA = copy(A)

    R = I(n)
    S = I(n)

    r = ones(n)
    s = ones(n)

    max_r = maximum(R) 
    max_s = maximum(S)

    while (max_r <= tol) && (max_s <= tol) 

        for i = 1:n

            r[i] = inv( sqrt( norm(@view(A[i, :]), p = Inf) ) )
            s[i] = inv( sqrt( norm(@view(A[:, i]), p = Inf) ) )

        end

        tmpA = diagm(r) * tmpA * diagm(s)
        R    = diagm(r) * R
        S    = S * diagm(s)

    end

    return R, S

end

function two_sided_diagonal_scaling(A, theta::AbstractFloat, tol::AbstractFloat)

    R, S = row_col_equilibration(A, tol)

    RAS = R * A * S

    beta = maximum(RAS)

    mu   = theta * prevfloat(typemax(Float16)) * inv(beta)

    return Float16.(mu .* RAS)

end


_determine_half(uL::Type{<:AbstractFloat}, uR::Type{<:AbstractFloat}) = any((uL, uR) .== Float16) ? Float16 : uL

_squeeze_into_half(A::AbstractMatrix, ::Type{Float16})         = two_sided_diagonal_scaling(A, 1.0, 0.4)
_squeeze_into_half(A::AbstractMatrix, ::Type{<:AbstractFloat}) = A

