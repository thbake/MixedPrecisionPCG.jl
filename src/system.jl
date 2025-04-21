using LinearAlgebra

export LinearSystem

mutable struct LinearSystem{T}

    A    ::AbstractMatrix{T}
    b    ::Vector{T}
    x    ::Vector{T}
    x0   ::Vector{T}
    normA::T
    normx::T

    function LinearSystem(
        A ::AbstractMatrix{T},
        b ::Vector{T},
        x ::Vector{T},
        x0::Vector{T}) where T

        new{T}(A, b, x, x0, norm(A), norm(x))

    end
    
end
