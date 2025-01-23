export hscg!, left_pcg!, right_pcg!, split_pcg!

function hscg!(convergence_data::ConvergenceData, A, b, x0, max_iter, tol = 1e-11)

    # Form initial residual and direction vectors

    x = x0
    r = b - A * x0
    p = r

    for k in 1:max_iter

        # A⋅pₖ₋₁
        Ap = A * p
        r_dot = dot(r, r)
        α = r_dot * inv( dot(p, Ap) ) 
        x = x + α .* p
        r = r - α .* Ap
        convergence_data.iterates[:, k] = x
        β = dot(r, r) * inv( r_dot )
        p = r + β .* p

    end

    return x

end

function left_pcg!(
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

        convergence_data.iterates[:, k] = x
        
        r     = r - α .* Ap
        z     = precondition(M, r)
        β     = dot(r, z) * inv(rz)
        p     = z + β .* p

    end

    return x
    
end

function right_pcg!(
    convergence_data::ConvergenceData,
    A               ::AbstractMatrix{uA}, 
    M               ::FactorizationPreconditioner{uL, uR, Right}, # Left preconditioner
    b               ::Vector{u}, # Right-hand side
    x0              ::Vector{u}, # Initial guess
    max_iter        ::Int,
    tol             ::AbstractFloat = 1e-11) where {u, uA, uL, uR}

    y = x0
    r = b - (A * uA.(x0))
    z = precondition(M,r)          # Usually a sparse triangular preconditioner.
    p = r
    #q = precondition(M,p)
    q = r

    for k = 1:max_iter
        q     = precondition(M, p)
        Aq    = A * uA.(q)
        rz    = dot(r, z)
        α     = rz * inv(dot(Aq, q))
        y     = y + α .* p

        convergence_data.iterates[:, k] = y
        
        r     = r - α .* Aq
        z     = precondition(M, r)
        β     = dot(r, z) * inv(rz)
        p     = r + β .* p

    end

    return y
    
end


"""
Split preconditioned CG.

Comments as "Performed in u" or "Stored in u" means in precision u denoted
by the corresponding unit roundoff / data type.
"""
function split_pcg!(
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
        β     = dot(r, r) * inv(rr)
        z     = precondition(M.Pr, r)
        p     = z + β .* p

    end

    return x
    
end
