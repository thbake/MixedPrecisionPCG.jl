export hscg!, left_pcg!

function hscg!(convergence_data::ConvergenceData, A, b, x_0, max_iter, tol = 1e-11)

    # Form initial residual and direction vectors

    x = x_0
    r = b - A * x_0
    p = r

    for k in 1:max_iter

        # A⋅pₖ₋₁
        u = A * p
        r_dot = dot(r, r)
        α = r_dot * inv( dot(p, u) ) 
        x = x + α .* p
        r = r - α .* u
        convergence_data.iterates[:, k] = x
        β = dot(r, r) * inv( r_dot )
        p = r + β .* p

    end

    return x

end

function left_pcg!(
    convergence_data::ConvergenceData,
    A               ::AbstractMatrix, 
    M               ::AbstractMatrix, # Left preconditioner
    b               ::AbstractVector, # Right-hand side
    x_0             ::AbstractVector, # Initial guess
    max_iter        ::Int,
    tol             ::AbstractFloat = 1e-11)

    x = x_0
    r = b - A * x_0
    z = M\r          # Usually a sparse triangular preconditioner.
    p = z

    for k = 1:max_iter

        u     = A * p
        r_dot = dot(r, z)
        α     = r_dot * inv(dot(u, p))
        x     = x + α * p
        convergence_data.iterates[:, k] = x
        r     = r - α * u
        z     = M\r
        β     = dot(r, z) * inv(r_dot)
        p     = z + β * p

    end

    return x
    
end
