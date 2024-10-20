export hscg!

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
