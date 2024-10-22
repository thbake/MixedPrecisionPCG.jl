@testset "Computation of error in the A-norm" begin

    n  = 3
    A  = SymTridiagonal(2ones(n), -ones(n - 1)) # SPD tridiagonal matrix
    x  = ones(n)  # Solution of linear system.
    b  = A * x    # Right-hand side.
    xk = 0.5 .* x # Approximate solution.

    # Error in the A-norm should be sqrt(0.5).
    @test A_norm(A, x - xk) == sqrt(0.5)
    

end
