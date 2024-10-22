export assemble_matrix, generate_eigs
export Spectrum, LeftSpectrum, RightSpectrum, EquallySpacedSpectrum
abstract type Spectrum end
struct LeftSpectrum          <: Spectrum end  
struct RightSpectrum         <: Spectrum end  
struct EquallySpacedSpectrum <: Spectrum end

function generate_eigs(λ1, λn, ρ, n, ::Spectrum)

    eigenvalues = [λ1 + ((i - 1) * inv(n - 1)) * (λn - λ1) * ρ^(n - i) for i in 2:n-1]

    return eigenvalues

end

function generate_eigs(λ1, λn, ρ, n, ::RightSpectrum)

    eigenvalues = [λn - ((i - 1) * inv(n - 1)) * (λn - λ1) * ρ^(n - i) for i in 2:n-1]

    return eigenvalues

end


function assemble_matrix(λ1::AbstractFloat, λn::AbstractFloat, ρ::AbstractFloat, n::Int, spectrum_type::Type{<:Spectrum})

    eigenvalues = vcat( λ1, generate_eigs(λ1, λn, ρ, n, spectrum_type()), λn )

    A = diagm(eigenvalues)

    return A

end
