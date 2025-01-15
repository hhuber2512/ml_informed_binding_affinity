using MCMCChains, Random, Distributions, CSV, DataFrames
include("calculate_experimental_outputs.jl")

function convert_to_linear_scale_from_logscale(parameter_array_logscale)
    p1 = 10.0.^parameter_array_logscale[1:end-1]
    p2 = exp(parameter_array_logscale[end])
    return append!(p1,p2)
end

function parameter_mapping(parameters)
    p = [:k_1 => parameters[1], :k_1inv => parameters[2], :k_2 => parameters[3], :k_3 => parameters[4], :k_4 => parameters[5], 
    :k_5 => parameters[6], :k_6 => parameters[7], :k_7 => parameters[8]]
    u0 = [:R => 10000.0, :L => 6.022E17, :RL => 0.0, :Gd => 3000.0, :Gbg => 3000.0, :G => parameters[9], :Ga => 0.0] 
    return u0, p
end

function parameter_mapping_dose_response(parameters, L)
    p = [:k_1 => parameters[1], :k_1inv => parameters[2], :k_2 => parameters[3], :k_3 => parameters[4], :k_4 => parameters[5], 
    :k_5 => parameters[6], :k_6 => parameters[7], :k_7 => parameters[8]]
    u0 = [:R => 10000.0, :L => L, :RL => 0.0, :Gd => 3000.0, :Gbg => 3000.0, :G => parameters[9], :Ga => 0.0] 
    return u0, p
end

function active_G_predictions(op, u0, p, solver_inputs)
    op = remake(op, u0=u0, p=p)
    predicted = DifferentialEquations.solve(op, solver_inputs["solver"], abstol=solver_inputs["abstol"], reltol=solver_inputs["reltol"], saveat=solver_inputs["saveat"]);
    fraction_activated = calculate_active_G_protein_fraction(predicted)
    return fraction_activated
end

function RL_predictions(op, u0, p, solver_inputs)
    op = remake(op, u0=u0, p=p)
    predicted = DifferentialEquations.solve(op, solver_inputs["solver"], abstol=solver_inputs["abstol"], reltol=solver_inputs["reltol"], saveat=solver_inputs["saveat"]);
    bound_receptor = predicted[:RL]
    return bound_receptor
end

function dose_response_predictions(op, u0, p, solver_inputs)
    op = remake(op, u0=u0, p=p)
    predicted = DifferentialEquations.solve(op, solver_inputs["solver"], abstol=solver_inputs["abstol"], reltol=solver_inputs["reltol"], saveat=solver_inputs["saveat"]);
    fraction_activated = calculate_active_G_protein_fraction(predicted)
    return fraction_activated
end

function sample_posterior_predictive_distribution(predictions, seed, std_dev)
    n = length(predictions)
    m = 10
    samples = Base.stack([rand(seed, Normal(predictions[i], std_dev), m) for i in 1:n])
    return reshape(samples,n*m)
end

function return_plot_inputs(case)
    if case == "binding_affinity_dose_response"
        x = DataFrame(CSV.File("data/active_G_dose_response.csv"))[!,"dose(log nM)"]
        response = deserialize("outputs/000_processed_$(case).dict")["response"]
        avg_error = deserialize("outputs/000_processed_$(case).dict")["average_error"]
        x_finegrain = collect(range(start=-2,stop=3,length=100))
        ylabel = "Receptor Affinity"
        xlabel = "log10[Alpha-Factor](nM)"
        xlims = [-2,3]
        ylims = [0,1.2]
    elseif case == "active_G_dose_response"
        x = DataFrame(CSV.File("data/active_G_dose_response.csv"))[!,"dose(log nM)"]
        response = deserialize("outputs/000_processed_$(case).dict")["response"]
        avg_error = deserialize("outputs/000_processed_$(case).dict")["average_error"]
        x_finegrain = collect(range(start=-2,stop=3,length=100))
        ylabel = "Active G Protein"
        xlabel = "log10[Alpha-Factor](nM)"
        xlims = [-2,3]
        ylims = [0,1.2]
    elseif case == "active_G_timecourse"
        x = deserialize("outputs/000_processed_$(case).dict")["save_at"]
        response = deserialize("outputs/000_processed_$(case).dict")["response"]
        avg_error = deserialize("outputs/000_processed_$(case).dict")["average_error"]
        x_finegrain = collect(range(start=0.0,stop=600.0, length=100))
        ylabel = "Active G Protein"
        xlabel = "time"
        xlims = [0,600]
        ylims = [0,0.5]
    end
    return ylims, xlims, xlabel, ylabel, x_finegrain,avg_error, response, x
end

function default_plot_settings(p,width)
    if width == 1
        formatted_plot = plot(p, dpi=400, size=(335,275), xguidefontsize=8, yguidefontsize=8,xtickfontsize=5,ytickfontsize=5,titlefontsize=12, 
        left_margin=[3Plots.mm 1Plots.mm],linewidth=4)
    elseif width == 2
        formatted_plot = plot(p, dpi=400, size=(690,500), xguidefontsize=8, yguidefontsize=8,xtickfontsize=5,ytickfontsize=5,titlefontsize=12, 
        left_margin=[3Plots.mm 1Plots.mm],linewidth=4)
    end
    return formatted_plot
end