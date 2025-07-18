using Plots, LaTeXStrings, MixedPrecisionPCG

export plot_accuracy_data, generate_plots, splitprec_comparison

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

function generate_plots(
    ad        ::AccuracyData,
    precisions::AbstractVector,
    scheme    ::Type{<:PreconditioningScheme},
    ylabel    ::Bool = true,
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
        #markersize = 2,
        #marker     = :circle,
        alpha      = 0.5,
        label      = labels,
        linestyle  = :solid,
        palette    = :Dark2_5,
        lw         = 1,
        legendfontsize = 7,
        ylabelfontsize = 6,
        xlabel = L"$k$"
        #xticks     = (1:49:200)
    )
    #scatter(x=(@view xs[1:1000:end]), y=(@view ys[1:1000:end]))
    #
    process_ylabel(string, ylabel::Bool) = ylabel ? string : ""

    p1 = plot(
        sample_data(ad.trueresnorm, step),
        #ad.trueresnorm,
        ylabel = process_ylabel(L"$\frac{||b - A\hat{x}_k||}{||A|| ||x||}$", ylabel),
        legend = !ylabel ? :topright : false
        
    )

    p2 = plot(
        sample_data(ad.updatedresnorm, step),
        ylabel = process_ylabel(L"$\frac{||\hat{r}_k||}{||A|| ||x||}$", ylabel),
        legend = :none
    )

    p3 = plot(
        sample_data(ad.errornorm, step),
        ylabel = process_ylabel(L"$\frac{||x - \hat{x}_k||_A}{||A||^{1/2} ||x||}$", ylabel),
        legend = :none
    )

    return p1, p2, p3

end

function plot_accuracy_data(plot_list, layout_matrix)

    p = plot(
        plot_list...,
        layout = layout_matrix
    )
    display(p)
        
end

function splitprec_comparison(ad_split, ad_split_saad, precisions)

    p1, p2, p3 = generate_plots(ad_split,      precisions, Split)
    p4, p5, p6 = generate_plots(ad_split_saad, precisions, SaadSplit, false)

    plot_list = [p1, p4, p2, p5, p3, p6] 

    plot_accuracy_data(plot_list, (3,2))

end

getexponent(number::AbstractFloat) = Int( floor( log10( number ) ) )


