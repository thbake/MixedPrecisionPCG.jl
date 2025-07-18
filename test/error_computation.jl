@testset "Computation of error in the A-norm" begin

    #n  = 3
    #A  = SymTridiagonal(2ones(n), -ones(n - 1)) # SPD tridiagonal matrix
    #x  = ones(n)            # Solution of linear system.
    #b  = A * x              # Right-hand side.
    #xk = 0.5 .* x           # Approximate solution.
    #exact_error = sqrt(0.5) # Exact error in the A-norm.

    #@test A_norm(A, x - xk) == exact_error

    #
    #max_iter = 3                                  # Let k = 2.
    #X        = stack( x - xk for _ in 1:max_iter) # nxk 
    #cd = ConvergenceData{Float64}(n, max_iter)
    #
    #A_norm!(cd, A, X)

    #@test cd.relative_error_norm == stack(exact_error for _ in 1:max_iter)
    

end
