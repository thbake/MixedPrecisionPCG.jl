using Plots, LaTeXStrings, MixedPrecisionPCG

export plot_accuracy_data, condition_plot

function getlabel(preconditioner::FactorizationPreconditioner{uL, uR, Left}) where {uL, uR}

    left_precision = getprecisions(preconditioner)

    return L"$u_L = $" * string(left_precision)  

end

function getlabel(preconditioner::FactorizationPreconditioner{uL, uR, Split}) where {uL, uR}

    left_precision, right_precision = getprecisions(preconditioner)
    return  L"$u_L = $" * string(left_precision) * L", $u_R = $" * string(right_precision) 

end

function getlabels(precisions::AbstractVector, label_dict, ::Split)

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

gettitle(::Left)  = "Mixed precision left PCG"
gettitle(::Split) = "Mixed precision split PCG"



resgapnorm_label(::Left)  = L"$\frac{||b - Ax_k - r_k||}{||A|| ||x||}$"

resgapnorm_label(::Split) = L"$\frac{||b - Ax_k - M_L \hat{r}_k||}{||A|| ||x||}$"

function plot_accuracy_data(
    ad        ::AccuracyData,
    precisions::AbstractVector,
    scheme    ::Type{<:PreconditioningScheme},
    step       ::Int = 1)

    label_dict = Dict(

        Float64 => "d",
        Float32 => "s",
        Float16 => "h"
    )

    labels = getlabels(precisions, label_dict, scheme())

    default(
        yscale     = :log10, 
        legend     = :bottomright,
        markersize = 2,
        marker = :circle,
        alpha      = 0.5,
        label      = labels,
        linestyle  = :solid,
        palette    = :Dark2_5,
        lw         = 1,
        legendfontsize = 7,
        ylabelfontsize = 6,
        #xticks     = (1:49:200)
    )
    #scatter(x=(@view xs[1:1000:end]), y=(@view ys[1:1000:end]))
    #

    p1 = plot(
        sample_data(ad.trueresnorm, step),
        #ad.trueresnorm,
        ylabel = L"$\frac{||b - Ax_k||}{||A|| ||x||}$",
        legend = false
        
    )

    p2 = plot(
        sample_data(ad.updatedresnorm, step),
        ylabel = L"$\frac{||r_k||}{||A|| ||x||}$",
        legend = false
    )

    p3 = plot(
        sample_data(ad.errornorm, step),
        ylabel = L"$\frac{||x - x_k||_A}{||x - x_0||_A}$",
        legend = false
    )

    p4 = plot(
        sample_data(ad.resgapnorm, step),
        ylabel = resgapnorm_label(scheme()),
        #yticks = 10.0 .^ collect(-17:2:0)
        yticks = :auto,
        legend = false
        
    )

    # Invisible plot for showing one legend
    invisible_data = [NaN for _ in 1:length(precisions)]'

    p5 = plot(
        invisible_data,
        legend     = true, 
        framestyle = :none
    )

    p6 = plot(
        invisible_data,
        legend     = false,
        framestyle = :none
    )

    p = plot(
        p1, p2, p5, p3, p4, p6, 
        layout = @layout([a b c{0.3w}; e d f{0.3w}]))

    display(p)

end

getexponent(number::AbstractFloat) = Int( floor( log10( number ) ) )

function condition_plot(
    ad_vec          ::Vector{AccuracyData},
    kappa_range     ::Vector{Float64},
    kappa_range_prec::Vector{Float64},
    precision       ::Type{<:AbstractFloat},
    n               ::Int)

    default(
        yscale   = :log10,
        legend   = :topright,
        ylabel   = L"$\frac{||b - A x_k - M_L \hat{r}_k ||}{||A|| ||x||}$"
    )

    p = plot()

    colors = palette(:Dark2_5)


    for i in eachindex(kappa_range)

        kappa_exponent = getexponent( kappa_range[i] )

        # Get number of steps that it takes the iterate to remain essentially unchanged.
        _, S = findmin(ad_vec[i].errornorm[1]) 

        # First (and in this case only) entry in the dictionary.
        max_ratio, max_ratio_idx = ad_vec[i].max_ratios[1]

        uL_kappa = upper_bound(kappa_range_prec[i], precision, Float64, n, S, max_ratio)

        label = raw"$U[\kappa(A)] \approx 10^{" * string( getexponent(uL_kappa) ) * raw"}, \;$" * raw"$\kappa(A) = 10^{" * string(kappa_exponent) * raw"}$" 
        println(kappa_range_prec[i])
        #label = raw"$\kappa(M_L) = 10^{" * string( log10(kappa_range[i]) ) * raw"}$"
        color = colors[i]

        plot!(
            ad_vec[i].resgapnorm,
            label   = "",
            ls      = :dot,
            lw      = 2,
            lc      = color,
            legend  = true
        )

        plot!(
        [uL_kappa for _ in 1:ad_vec[i].iter_number], 
            linestyle   = :solid,
            label       = label,
            linecolor   = color
        )

        scatter!(
            [max_ratio_idx], 
            [1.0],
            marker      = :cross,
            markercolor = color,
            label       = "",
        )


    end

    display(p)

end

#scatter!([idx[2]], [1.0], marker = :xcross, markercolor = :black, label = L"$max_j ||x_j||$")
