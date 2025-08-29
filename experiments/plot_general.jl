using Plots, LaTeXStrings, MixedPrecisionPCG, BFloat16s

export plot_accuracy_data, generate_plots, splitprec_comparison, get_ylabel,
       plot_eigenvalues, TrueResNorm, UpdatedResNorm, ErrorNorm
    

function getlabel(preconditioner::FactorizationPreconditioner{uL, uR, Left}) where {uL, uR}

    left_precision = getprecisions(preconditioner)

    return L"$u_L = $" * string(left_precision)  

end

function getlabel(preconditioner::FactorizationPreconditioner{uL, uR, AbstractSplit}) where {uL, uR}

    left_precision, right_precision = getprecisions(preconditioner)
    return  L"$u_L = $" * string(left_precision) * L", $u_R = $" * string(right_precision) 

end

function getlabels(precisions::AbstractVector, label_dict, ::AbstractSplit)

    labels = Vector{String}(undef, length(precisions))

    i = 1

    for (prec1, prec2) in precisions

        
        label = "(" * label_dict[prec1] * "," * label_dict[prec2] * ")"

        labels[i] = label

        i += 1

    end

    return permutedims(labels)

end

function getlabels(precisions::AbstractVector, label_dict, ::Left)

    labels = Vector{String}(undef, length(precisions))

    for i in eachindex(precisions)

        labels[i] = label_dict[precisions[i]]

    end

    return permutedims(labels)
end


function sample_data(data::Vector{<:AbstractVector}, step::Int)

    data_cpy = deepcopy(data)

    for data_set in data_cpy

        for j in eachindex(data_set)

            if j % step != 0

                data_set[j] = NaN

            end

        end

    end

    return data_cpy

    #data = [data[i][1:step:end] for i in eachindex(data)]

end

gettitle(::Left)      = "Mixed precision left PCG"
gettitle(::Split)     = "Mixed precision split PCG"
gettitle(::SaadSplit) = "Mixed precision Saad's split PCG"



resgapnorm_label(::Left)  = L"$\frac{||b - Ax_k - r_k||}{||A|| ||x||}$"

resgapnorm_label(::AbstractSplit) = L"$\frac{||b - Ax_k - M_L \hat{r}_k||}{||A|| ||x||}$"


# Wrappers
# ===================
abstract type AbstractWrapper end

Base.length(gw::AbstractWrapper)               = length(aw.ads)
Base.getindex(gw::AbstractWrapper, i::Integer) = aw.ads[i]
Base.eachindex(gw::AbstractWrapper)            = eachindex(aw.ads)

struct TrueResNorm <: AbstractWrapper
    ads::AccuracyDataSeries
end

struct UpdatedResNorm <: AbstractWrapper
    ads::AccuracyDataSeries
end

struct ErrorNorm <: AbstractWrapper
    ads::AccuracyDataSeries
end

struct GenericWrapper <:AbstractWrapper
    ads::AccuracyDataSeries
end

@recipe trn(::Type{TrueResNorm},    trn::TrueResNorm)    = [trn.ads[i].trueresnorm    for i in eachindex(trn.ads)]
@recipe urn(::Type{UpdatedResNorm}, urn::UpdatedResNorm) = [urn.ads[i].updatedresnorm for i in eachindex(urn.ads)]
@recipe  en(::Type{ErrorNorm},       en::ErrorNorm)      = [ en.ads[i].errornorm      for i in eachindex( en.ads)]
@recipe  gw(::Type{GenericWrapper},  gw::GenericWrapper) = TrueResNorm(gw.ads)

get_ylabel(::Type{TrueResNorm})    = L"$\frac{||b - A\hat{x}_k||}{||A|| ||x||}$"
get_ylabel(::Type{UpdatedResNorm}) = L"$\frac{||\hat{r}_k||}{||A|| ||x||}$"
get_ylabel(::Type{ErrorNorm})      = L"$\frac{||x - \hat{x}_k||_A}{||A||^{1/2} ||x||}$"


@userplot AccuracyPlot
@recipe function f(
    ap          ::AccuracyPlot;
    precisions  ::AbstractVector,
    scheme      ::Type{<:PreconditioningScheme},
    legend_param = true::Bool,
    ylabel_param = true::Bool,
    upperbound  ::Float64)
   
    # Generate labels
    label_dict = Dict(

        Float64  => "d",
        Float32  => "s",
        Float16  => "h",
        BFloat16 => "b16"
    )

    # Extract y from the args
    y = ap.args

    alpha          --> 0.9
    labels         --> cat([""], getlabels(precisions, label_dict, scheme()); dims = 2)
    legend         := legend_param ? :topright : :none
    legendfontsize --> 7
    linestyle      --> :solid
    xlabel         --> L"$k$"
    ylabelfontsize --> 6
    ylabel         := ylabel_param ? get_ylabel(eltype(y)) : ""
    yscale         --> :log10

    @series begin
        linestyle := :dash
        #label --> L"n k^2 u \kappa(M)^{1/2}"
        label --> "bound"
        legend := :none
        repeat([upperbound], get_iternumber(y[1].ads))
    end
    return y 

end

function compplot(
    ads_pair   ::Tuple{AccuracyDataSeries, AccuracyDataSeries},
    prec_pair  ::Tuple{AbstractVector, AbstractVector}, 
    scheme_pair::Tuple{T, T},
    bound_pair ::Tuple{Float64, Float64}) where T <: Type

    layout = (3,2)

    p = Vector{Plots.Plot{Plots.GRBackend}}(undef, 6)
    
    i = 0

    get_upperbound(::Type{<:AbstractWrapper}) = bound_pair[1]
    get_upperbound(::Type{ErrorNorm})         = bound_pair[2]

    for (ads, prec, scheme) in zip(ads_pair, prec_pair, scheme_pair)

        i += 1

        bool_ylabel = Bool(1 % i)

        for (j, wrapper) in enumerate((TrueResNorm, UpdatedResNorm, ErrorNorm))

            bool_legend = Bool(1 % j)

            p[i + (j - 1) * (3 - 1) ] = accuracyplot(
                wrapper(ads),
                precisions   = prec,
                scheme       = scheme,
                legend_param = !bool_legend,
                ylabel_param = !bool_ylabel,
                upperbound   = get_upperbound(wrapper))

        end

    end

    plot(p..., layout = layout)
end

function f(
    ads1       ::AccuracyDataSeries,
    ads2       ::AccuracyDataSeries,
    prec_pair  ::Tuple{AbstractVector, AbstractVector}, 
    scheme_pair::Tuple{T, T}) where T <: Type

    #println("hello")
    
    
    #layout := (1,2)
    #@series begin
    #    accuracyplot(ErrorNorm(ads1), precisions = prec_pair[1], scheme = scheme_pair[1])
    #end
    #@series begin
    #    subplot := 2
    #    accuracyplot(ErrorNorm(ads2), precisions = prec_pair[2], scheme = scheme_pair[2])
    #end
    p1, p2 = prec_pair
    s1, s2 = scheme_pair
    #accuracyplot(ErrorNorm(ads1), precisions = prec_pair[1], scheme = scheme_pair[1])
    @series begin
        return accuracyplot(ErrorNorm(ads1), precisions = p1, scheme = s1)
    end
end


getexponent(number::AbstractFloat) = Int( floor( log10( number ) ) )

function plot_eigenvalues(A)

    eigs = eigvals(Matrix(A))
    
    ploteigs = [(λ, 0.0) for λ in eigs]

    l = @layout [a;b]

    p1 = histogram(eigs, palette = cgrad(:blues).colors, yscale = :log10, bins = :scott)

    p2 = plot(
        ploteigs, 
        marker      = :xcross,
        markercolor = :red,
        linecolor   = :black,
        ylims       = (-1,1),
        xscale       = :identity,
        label       = L"$\lambda \in \Lambda (A)$")

    p = plot(p1, p2, layout = l)

    display(p)
end




