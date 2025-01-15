using Serialization, MCMCChains, Plots

"""

save\\_rhat\\_ess\\_iteration(mcmc\\_chain: MCMCChains.Chains, n\\_iter\\_divisions:Int64, approach: String, ensemble: Int64, holdoneout: String)

Function calculates r-hat and effective sample size for n\\_iter\\_divisions of the complete mcmc\\_chain. \n 
Provides idea of how r-hat is changing as posterior samples increase \n

Should *save*: \n
Dict{String, Array{Float64}} with 3 entries. For example: \n
"ess"       => [4162.91 3924.33 … 5652.56 5884.53; 8273.19 7775.98 … 10857.1 … \n
"iteration" => [1100.0, 2200.0, 3300.0, 4400.0, 5500.0, 6600.0, 7700.0, 8800.… \n
"rhat"      => [1.32845 1.37226 … 1.22909 1.20705; 1.18661 1.22845 … 1.15737 \n

ess Array: n\\_iter\\_divisions x n\\_parameters \n
iteration Array: n\\_iter\\_divisions \n
rhat Array: n\\_iter\\_divisions x n\\_parameters \n

"""
function save_rhat_ess_iteration(mcmc_chain, n_iter_divisions, approach, ensemble)
    samples = Base.stack(Array(mcmc_chain, append_chains=false)); #convert mcmc chain to 3D array (iteration x parameters x chains)
    n_iterations = size(samples)[1]
    n_parameters = size(samples)[2]
    n_chains = size(samples)[3]
    interval = n_iterations/n_iter_divisions
    rhats = Array{Float64}(undef, n_iter_divisions, n_parameters)
    ess = Array{Float64}(undef, n_iter_divisions, n_parameters)
    iterations_vect = Array{Float64}(undef, n_iter_divisions)
    for i=1:n_iter_divisions
        range_max = Int(interval*i)
        convert_chains = Chains(samples[1:range_max,:,:])
        diagnostics = MCMCChains.ess_rhat(convert_chains)
        rhats[i,:] = diagnostics[:,:rhat] #store rhat
        ess[i,:] = diagnostics[:,:ess] #store ess
        iterations_vect[i] = range_max 
    end
    diagnostics_dictionary = Dict("iteration"=>iterations_vect, "rhat"=>rhats, "ess"=>ess)
    serialize("outputs/005_ess_rhats_$(approach)_ensemble$(ensemble).jls", diagnostics_dictionary)
end

"""

plot\\_rhats(diagnostics\\_dict\\_list\\_cntrl: Vector{Dict{String, Array{Float64}}} of length n_ensemble, diagnostics\\_dict\\_list\\_exp: Vector{Dict{String, Array{Float64}}} of length n_ensemble, n_ensemble: Int64)

Function plots r-hat vs. posterior sample size for each of 9 parameters \n 
Both control and experimental condition are plotted \n
Plots are saved to current path. \n

"""
function plot_rhats(diagnostics_dict_list_cntrl, diagnostics_dict_list_exp, n_ensemble)
    parameter_string_contrl = [ "k_1", "kinv", "k_2", "k_3", "k_4", "k_5", "k_6", "k_7", "G"]
    parameter_string_exp = [ "k_1", "kinv", "k_2", "k_3", "k_4", "k_5", "k_6", "k_7", "G"]
    index = 1
    p1 = plot()
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_cntrl[j]
        if j == 1
            plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], label="control", linecolor=:darkturquoise, linewidth=4)
        else
            plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], label=false, linecolor=:darkturquoise, linewidth=4)
        end
    end
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_exp[j]
        if j == 1
            plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], label="experimental", linecolor=:deeppink4, linewidth=4)
        else
            plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], label=false, linecolor=:deeppink4, linewidth=4)
        end
    end
    ylabel!("r-hat $(parameter_string_contrl[index])")
    xlabel!("iteration")
    hline!([1.1], linecolor=:red3, linewidth=4, label="convergence")
    
    index = 2
    p2 = plot()
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_cntrl[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], legend=false, linecolor=:darkturquoise, linewidth=4)
    end
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_exp[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], legend=false, linecolor=:deeppink4, linewidth=4)
    end
    ylabel!("r-hat $(parameter_string_contrl[index])")
    hline!([1.1], linecolor=:red3, linewidth=4, label="convergence")

    index = 3
    p3 = plot()
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_cntrl[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], legend=false, linecolor=:darkturquoise, linewidth=4)
    end
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_exp[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], legend=false, linecolor=:deeppink4, linewidth=4)
    end
    ylabel!("r-hat $(parameter_string_contrl[index])")
    hline!([1.1], linecolor=:red3, linewidth=4, label="convergence")

    index = 4
    p4 = plot()
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_cntrl[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], legend=false, linecolor=:darkturquoise, linewidth=4)
    end
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_exp[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], legend=false, linecolor=:deeppink4, linewidth=4)
    end
    ylabel!("r-hat $(parameter_string_contrl[index])")
    hline!([1.1], linecolor=:red3, linewidth=4, label="convergence")

    index = 5
    p5 = plot()
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_cntrl[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], legend=false, linecolor=:darkturquoise, linewidth=4)
    end
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_exp[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], legend=false, linecolor=:deeppink4, linewidth=4)
    end
    ylabel!("r-hat $(parameter_string_contrl[index])")
    hline!([1.1], linecolor=:red3, linewidth=4, label="convergence")

    index = 6
    p6 = plot()
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_cntrl[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], legend=false, linecolor=:darkturquoise, linewidth=4)
    end
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_exp[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], legend=false, linecolor=:deeppink4, linewidth=4)
    end
    ylabel!("r-hat $(parameter_string_contrl[index])")
    hline!([1.1], linecolor=:red3, linewidth=4, label="convergence")

    index = 7
    p7 = plot()
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_cntrl[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], legend=false, linecolor=:darkturquoise, linewidth=4)
    end
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_exp[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], legend=false, linecolor=:deeppink4, linewidth=4)
    end
    ylabel!("r-hat $(parameter_string_contrl[index])")
    hline!([1.1], linecolor=:red3, linewidth=4, label="convergence")

    index = 8
    p8 = plot()
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_cntrl[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], legend=false, linecolor=:darkturquoise, linewidth=4)
    end
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_exp[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], legend=false, linecolor=:deeppink4, linewidth=4)
    end
    ylabel!("r-hat $(parameter_string_contrl[index])")
    hline!([1.1], linecolor=:red3, linewidth=4, label="convergence")

    index = 9
    p9 = plot()
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_cntrl[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], legend=false, linecolor=:darkturquoise, linewidth=4)
    end
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_exp[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["rhat"][:,index], legend=false, linecolor=:deeppink4, linewidth=4)
    end
    ylabel!("r-hat $(parameter_string_contrl[index])")
    hline!([1.1], linecolor=:red3, linewidth=4, label="convergence")

    p_final = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, ylims=(1,1.3), layout=(3,3), dpi=300, size=(900,800))
    return p_final
end

"""

plot\\_ess(diagnostics\\_dict\\_list\\_cntrl: Vector{Dict{String, Array{Float64}}} of length n_ensemble, diagnostics\\_dict\\_list\\_exp: n_ensemble Vector{Dict{String, Array{Float64}}} of length n_ensemble, n_ensemble: Int64)

Function plots ess vs. posterior sample size for each of 9 parameters \n 
Both control and experimental condition are plotted \n
Should return: \n
Plot of ess: Plot object

"""
function plot_ess(diagnostics_dict_list_cntrl, diagnostics_dict_list_exp, n_ensemble, n_walkers)
    minimum_ess = 100*n_walkers
    parameter_string_contrl = [ "k_1", "kinv", "k_2", "k_3", "k_4", "k_5", "k_6", "k_7", "G"]
    parameter_string_exp = [ "k_1", "kinv", "k_2", "k_3", "k_4", "k_5", "k_6", "k_7", "G"]
    index = 1
    p1 = plot()
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_cntrl[j]
        if j == 1
            plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], label="control", linecolor=:darkturquoise, linewidth=4)
        else
            plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], label=false, linecolor=:darkturquoise, linewidth=4)
        end
    end
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_exp[j]
        if j == 1
            plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], label="experimental", linecolor=:deeppink4, linewidth=4)
        else
            plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], label=false, linecolor=:deeppink4, linewidth=4)
        end
    end
    ylabel!("ess $(parameter_string_contrl[index])")
    xlabel!("iteration")
    hline!([minimum_ess], linecolor=:red3, linewidth=4, label="minimum")
    
    index = 2
    p2 = plot()
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_cntrl[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], legend=false, linecolor=:darkturquoise, linewidth=4)
    end
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_exp[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], legend=false, linecolor=:deeppink4, linewidth=4)
    end
    ylabel!("ess $(parameter_string_contrl[index])")
    hline!([minimum_ess], linecolor=:red3, linewidth=4, label="minimum")

    index = 3
    p3 = plot()
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_cntrl[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], legend=false, linecolor=:darkturquoise, linewidth=4)
    end
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_exp[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], legend=false, linecolor=:deeppink4, linewidth=4)
    end
    ylabel!("ess $(parameter_string_contrl[index])")
    hline!([minimum_ess], linecolor=:red3, linewidth=4, label="minimum")

    index = 4
    p4 = plot()
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_cntrl[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], legend=false, linecolor=:darkturquoise, linewidth=4)
    end
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_exp[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], legend=false, linecolor=:deeppink4, linewidth=4)
    end
    ylabel!("ess $(parameter_string_contrl[index])")
    hline!([minimum_ess], linecolor=:red3, linewidth=4, label="minimum")

    index = 5
    p5 = plot()
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_cntrl[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], legend=false, linecolor=:darkturquoise, linewidth=4)
    end
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_exp[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], legend=false, linecolor=:deeppink4, linewidth=4)
    end
    ylabel!("ess $(parameter_string_contrl[index])")
    hline!([minimum_ess], linecolor=:red3, linewidth=4, label="minimum")

    index = 6
    p6 = plot()
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_cntrl[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], legend=false, linecolor=:darkturquoise, linewidth=4)
    end
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_exp[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], legend=false, linecolor=:deeppink4, linewidth=4)
    end
    ylabel!("ess $(parameter_string_contrl[index])")
    hline!([minimum_ess], linecolor=:red3, linewidth=4, label="minimum")

    index = 7
    p7 = plot()
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_cntrl[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], legend=false, linecolor=:darkturquoise, linewidth=4)
    end
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_exp[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], legend=false, linecolor=:deeppink4, linewidth=4)
    end
    ylabel!("ess $(parameter_string_contrl[index])")
    hline!([minimum_ess], linecolor=:red3, linewidth=4, label="minimum")

    index = 8
    p8 = plot()
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_cntrl[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], legend=false, linecolor=:darkturquoise, linewidth=4)
    end
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_exp[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], legend=false, linecolor=:deeppink4, linewidth=4)
    end
    ylabel!("ess $(parameter_string_contrl[index])")
    hline!([minimum_ess], linecolor=:red3, linewidth=4, label="minimum")

    index = 9
    p9 = plot()
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_cntrl[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], legend=false, linecolor=:darkturquoise, linewidth=4)
    end
    for j=1:n_ensemble
        diagnostics_dictionary = diagnostics_dict_list_exp[j]
        plot!(diagnostics_dictionary["iteration"], diagnostics_dictionary["ess"][:,index], legend=false, linecolor=:deeppink4, linewidth=4)
    end
    ylabel!("ess $(parameter_string_contrl[index])")
    hline!([minimum_ess], linecolor=:red3, linewidth=4, label="minimum")

    p_final = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, layout=(3,3), dpi=300, size=(1000,1000), xaxis=(xtickfontrotation=20), yaxis=(ytickfontrotation=20))
    return p_final
end