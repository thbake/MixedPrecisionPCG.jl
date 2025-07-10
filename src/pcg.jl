export hscg!, pcg!

function hscg!(convergence_data::ConvergenceData, A, b, x0, max_iter, tol = 1e-11)

    # Form initial residual and direction vectors

    x = x0
    r = b - (A * x0)
    p = r

    for k in 1:max_iter

        # A⋅pₖ₋₁
        Ap = A * p
        r_dot = dot(r, r)
        α = r_dot /  dot(p, Ap)  
        x = x + α .* p
        r = r - α .* Ap
        convergence_data.iterates[:, k]          = x
        convergence_data.updated_residuals[:, k] = r
        β = dot(r, r) / ( r_dot )
        p = r + (β .* p)

    end

    return x

end

"""
Left preconditioned CG.

The algorithm is mathematically and numerically equivalent to right 
preconditioned CG.
"""
function pcg!(
    convergence_data::ConvergenceData,
    A               ::AbstractMatrix{uA}, 
    M               ::FactorizationPreconditioner{uL, uR, Left}, # Left preconditioner
    b               ::Vector{u}, # Right-hand side
    x0              ::Vector{u}, # Initial guess
    max_iter        ::Int,
    tol             ::AbstractFloat = 1e-11) where {u, uA, uL, uR}

    x = x0
    r = b - (A * uA.(x0))
    z = precondition(M, r)          # Usually a sparse triangular preconditioner.
    p = z

    for k = 1:max_iter

        Ap    = A * uA.(p)
        rz    = dot(r, z)
        α     = rz * inv(dot(Ap, p))
        x     = x + α .* p
        r     = r - α .* Ap

        convergence_data.iterates[:, k]          = x
        convergence_data.updated_residuals[:, k] = r

        z     = precondition(M, r)
        β     = dot(r, z) * inv(rz)
        p     = z + β .* p

    end

    return x
    
end


"""
Split preconditioned CG [Algorithm 9.2,Saad 2003].

Comments as "Performed in u" or "Stored in u" means in precision u denoted
by the corresponding unit roundoff / data type.
"""
function pcg!(
    convergence_data::ConvergenceData,
    A               ::AbstractMatrix{uA}, 
    M               ::FactorizationPreconditioner{uL, uR, Split},
    b               ::Vector{u}, # Right-hand side
    x0              ::Vector{u}, # Initial guess
    max_iter        ::Int,
    tol             ::AbstractFloat = 1e-11) where {u, uA, uL, uR}

    x = x0
    r = b - (A * uA.(x0))
    p = precondition!(M, r)

    for k in 1:max_iter

        Ap    = A * uA.(p)               # Multiply in uA and store in u.
        rr    = dot(r, r)                # Performed in u.
        α     = rr * inv(dot(Ap, p))     # Performed in u.
        x     = x + α .* p               # Update in u.

        convergence_data.iterates[:, k] = x

        v     = precondition(M.Pl, Ap)   # Apply left preconditioner in uL and store in u.
        r     = r - α .* v

        convergence_data.updated_residuals[:, k] = r
        β     = dot(r, r) * inv(rr)
        z     = precondition(M.Pr, r)
        p     = z + β .* p

    end

    return x
    
end

function pcg!(
    convergence_data::ConvergenceData,
    A               ::AbstractMatrix{uA},
    M               ::FactorizationPreconditioner{uL, uR,<:PreconditioningScheme},
    b               ::Vector{u},
    x0              ::Vector{u},
    max_iter        ::Int, 
    scheme          ::PreconditioningScheme) where {u, uA, uL, uR}

    x = x0
    r = b - (A * uA.(x0))


    s, q, z = general_precond(M, r)

    p = q

    for k in 1:max_iter

        Ap = A * uA.(p)
        zs = dot(z, s)
        α  = zs * inv(dot(Ap, p))
        x  = x + α .* p

        convergence_data.iterates[:, k] = x

        r = r - α .* Ap

        convergence_data.updated_residuals[:, k] = r

        s, q, z = general_precond(M, r)

        β = dot(z,s) * inv(zs)
        p = q + β .* p

    end

    return x

end

    
