using Turing, Distributions, Catalyst, DifferentialEquations, DynamicPPL
include("regularization.jl")
include("calculate_experimental_outputs.jl")

"""

logprob\\_regularized(data: Vector{Float64}, odemodel: DifferentialEquations.ODEProblem, σ: Vector{Float64})

Defines target distribution for posterior sampling using Turing @model macro. \n
Target distribution includes extra regularization term. \n

Parameters: \n
data: Vector{Float64}, 1D vector of n datapoints, here, concatenated timecourse and dose response data \n
odemodel: DifferentialEquations.ODEProblem, gpcr ode model for simulation \n
σ: Vector{Float64}, 1D vector of n standard deviations associated with each data points\n

Should return: \n
joint loglikelihood, type: DynamicPPL.Model

"""

Turing.@model function logprob_regularized(data, odeproblem, σ, regularization, odesolver_timecourse, dosages)
    
    #define prior distributions for parameters we will infer; forward and reverse reaction rates
    #units of concentration are nM
    k5b_est ~ Uniform(-6,0) #0.2 
    k5f_est ~ Uniform(-4,0) #0.06
    k13b_est ~ Uniform(-6,0) #0.6
    k13f_est ~ Uniform(-4,0) #0.09  
    k21f_est ~ Uniform(-4,0) #0.003 
    k21b_est ~ Uniform(-6,0) #0.1 
    k1f_est ~ Uniform(-4,0) #0.003
    k1b_est ~ Uniform(-6,0) #0.06   
    k12f_est ~ Uniform(-4,0) #0.0015
    k12b_est ~ Uniform(-6,0) #0.0001  
    k19f_est ~ Uniform(-4,0) #0.01
    k19b_est ~ Uniform(-6,0) #0.0214 
    k17f_est ~ Uniform(-4,0) #0.003
    k17b_est ~ Uniform(-6,0) #0.1 
    k9b_est ~ Uniform(-6,0) #0.05    
    k9f_est ~ Uniform(-4,0) #0.003 
    k10f_est ~ Uniform(-4,0) #0.01 
    k10b_est ~ Uniform(-6,0) #0.06

    #regularization factor
    @addlogprob! regularize(regularization, k5b_est, k5f_est, "k5") #regularization based on ratio of k5
    @addlogprob! regularize(regularization, k13b_est, k13f_est, "k13") #regularization based on ratio of k13
    @addlogprob! regularize(regularization, k21b_est, k21f_est, "k21") #regularization based on ratio of k21
    @addlogprob! regularize(regularization, k1b_est, k1f_est, "k1") #regularization based on ratio of k1
    @addlogprob! regularize(regularization, k12b_est, k12f_est,"k12") #regularization based on ratio of k12
    @addlogprob! regularize(regularization, k19b_est, k19f_est,"k19") #regularization based on ratio of k19
    @addlogprob! regularize(regularization, k17b_est, k17f_est,"k17") #regularization based on ratio of k17 
    @addlogprob! regularize(regularization, k9b_est, k9f_est,"k9") #regularization based on ratio of k9
    @addlogprob! regularize(regularization, k10b_est, k10f_est,"k10") #regularization based on ratio of k10
    
    p_new = 
    [:k7b => 0.006, :k22f => 0.03, :k23b => 0.021, :k22b => 0.064, :k20f => 0.12, :k24b => 0.0429, :k14f => 6.0, :k25b => 0.03, 
    :k5b => 10^(k5b_est), :k13b => 10^(k13b_est), :k23f => 0.1, :k7f => 0.3, :k21f => 10^(k21f_est), :k11b => 0.0045, :k21b => 10^(k21b_est), 
    :V4 => 450.0, :k1f => 10^(k1f_est), :k18b => 0.0009, :k12f => 10^(k12f_est), :k25f => 1.0, :k19f => 10^(k19f_est), :k24f => 0.009, 
    :k6f => 1.0, :k3f => 1.0, :k17f => 10^(k17f_est), :k9b => 10^(k9b_est), :K16 => 340.0, :V8 => 1.0, :k11f => 0.03, :k14b => 0.06, 
    :k6b => 0.05, :k5f => 10^(k5f_est), :k15b => 0.0009, :k19b => 10^(k19b_est), :k13f => 10^(k13f_est), :k1b => 10^(k1b_est), 
    :k12b => 10^(k12b_est), :K4 => 50.0, :K8 => 100.0, :k17b => 10^(k17b_est), :ADP => 1.0, :k3b => 0.01, :k9f => 10^(k9f_est), :ATP => 1.0,
    :k20b => 0.00024, :k10f => 10^(k10f_est), :V16 => 1.7, :k15f => 0.3, :k2b => 0.1, :k18f => 0.3, :k2f => 0.01, :k10b => 10^(k10b_est),
    :default_compartment => 1.0] 
    
    #redefine ode problem with sampled values
    #per SciML docs, we make sure to input parameter map, rather than rely on parameter order
    op = remake(odeproblem, p=p_new)

    #timecourse simulation
    save_at = odesolver_timecourse["saveat"]
    solver = odesolver_timecourse["solver"]
    abstol = odesolver_timecourse["abstol"]
    reltol = odesolver_timecourse["reltol"]

    #response to 20nM EGF
    egf = dosages[1]
    odesys, u0, tspan, p = return_ode_problem_default_inputs(egf)
    op = remake(op, u0=u0)
    predicted = DifferentialEquations.solve(op, solver, abstol=abstol, reltol=reltol, saveat=save_at);
    #Early exit if simulation could not be computed successfully.
    if predicted.retcode !== ReturnCode.Success
        Turing.@addlogprob! -Inf
        return
    end
    experimental_quantities = calculate_all_quantities(predicted)
    species_index = return_index_order_of_data_for_likelihood(egf)
    all_predictions = Array{Float64}(undef,0) #make sure type works as input for MvNormal
    [append!(all_predictions, experimental_quantities[i]) for i in species_index];

    #response to 2 EGF
    egf = dosages[2]
    odesys, u0, tspan, p = return_ode_problem_default_inputs(egf)
    op = remake(op, u0=u0)
    predicted = DifferentialEquations.solve(op, solver, abstol=abstol, reltol=reltol, saveat=save_at);
    #Early exit if simulation could not be computed successfully.
    if predicted.retcode !== ReturnCode.Success
        Turing.@addlogprob! -Inf
        return
    end
    experimental_quantities = calculate_all_quantities(predicted)
    species_index = return_index_order_of_data_for_likelihood(egf)
    [append!(all_predictions, experimental_quantities[i]) for i in species_index];

    #response to 0.2 EGF
    egf = dosages[3]
    odesys, u0, tspan, p = return_ode_problem_default_inputs(egf)
    op = remake(op, u0=u0)
    predicted = DifferentialEquations.solve(op, solver, abstol=abstol, reltol=reltol, saveat=save_at);
    #Early exit if simulation could not be computed successfully.
    if predicted.retcode !== ReturnCode.Success
        Turing.@addlogprob! -Inf
        return
    end
    experimental_quantities = calculate_all_quantities(predicted)
    species_index = return_index_order_of_data_for_likelihood(egf)
    [append!(all_predictions, experimental_quantities[i]) for i in species_index];

    #data likelihood
    data ~ MvNormal(all_predictions, σ) # MvNormal can take a vector input for standard deviation, and assumes covariance matrix is diagonal
end

"""

logprob\\_unregularized(data: Vector{Float64}, odemodel: DifferentialEquations.ODEProblem, σ: Vector{Float64})

Defines target distribution for posterior sampling using Turing @model macro. \n

Parameters: \n
data: Vector{Float64}, 1D vector of n datapoints, here, concatenated timecourse and dose response data \n
odemodel: DifferentialEquations.ODEProblem, gpcr ode model for simulation \n
σ: Vector{Float64}, 1D vector of n standard deviations associated with each data points\n

Should return: \n
joint loglikelihood, type: DynamicPPL.Model

"""
Turing.@model function logprob_unregularized(data, odeproblem, σ, odesolver_timecourse, dosages)
    
    #define prior distributions for parameters we will infer; forward and reverse reaction rates
    #units of concentration are nM
    k5b_est ~ Uniform(-6,0) #0.2 
    k5f_est ~ Uniform(-4,0) #0.06
    k13b_est ~ Uniform(-6,0) #0.6
    k13f_est ~ Uniform(-4,0) #0.09  
    k21f_est ~ Uniform(-4,0) #0.003 
    k21b_est ~ Uniform(-6,0) #0.1 
    k1f_est ~ Uniform(-4,0) #0.003
    k1b_est ~ Uniform(-6,0) #0.06   
    k12f_est ~ Uniform(-4,0) #0.0015
    k12b_est ~ Uniform(-6,0) #0.0001  
    k19f_est ~ Uniform(-4,0) #0.01
    k19b_est ~ Uniform(-6,0) #0.0214 
    k17f_est ~ Uniform(-4,0) #0.003
    k17b_est ~ Uniform(-6,0) #0.1 
    k9b_est ~ Uniform(-6,0) #0.05    
    k9f_est ~ Uniform(-4,0) #0.003 
    k10f_est ~ Uniform(-4,0) #0.01 
    k10b_est ~ Uniform(-6,0) #0.06
    
    p_new = 
    [:k7b => 0.006, :k22f => 0.03, :k23b => 0.021, :k22b => 0.064, :k20f => 0.12, :k24b => 0.0429, :k14f => 6.0, :k25b => 0.03, 
    :k5b => 10^(k5b_est), :k13b => 10^(k13b_est), :k23f => 0.1, :k7f => 0.3, :k21f => 10^(k21f_est), :k11b => 0.0045, :k21b => 10^(k21b_est), 
    :V4 => 450.0, :k1f => 10^(k1f_est), :k18b => 0.0009, :k12f => 10^(k12f_est), :k25f => 1.0, :k19f => 10^(k19f_est), :k24f => 0.009, 
    :k6f => 1.0, :k3f => 1.0, :k17f => 10^(k17f_est), :k9b => 10^(k9b_est), :K16 => 340.0, :V8 => 1.0, :k11f => 0.03, :k14b => 0.06, 
    :k6b => 0.05, :k5f => 10^(k5f_est), :k15b => 0.0009, :k19b => 10^(k19b_est), :k13f => 10^(k13f_est), :k1b => 10^(k1b_est), 
    :k12b => 10^(k12b_est), :K4 => 50.0, :K8 => 100.0, :k17b => 10^(k17b_est), :ADP => 1.0, :k3b => 0.01, :k9f => 10^(k9f_est), :ATP => 1.0,
    :k20b => 0.00024, :k10f => 10^(k10f_est), :V16 => 1.7, :k15f => 0.3, :k2b => 0.1, :k18f => 0.3, :k2f => 0.01, :k10b => 10^(k10b_est),
    :default_compartment => 1.0] 
    
    #redefine ode problem with sampled values
    #per SciML docs, we make sure to input parameter map, rather than rely on parameter order
    op = remake(odeproblem, p=p_new)

    #timecourse simulation
    save_at = odesolver_timecourse["saveat"]
    solver = odesolver_timecourse["solver"]
    abstol = odesolver_timecourse["abstol"]
    reltol = odesolver_timecourse["reltol"]

    #response to 20nM EGF
    egf = dosages[1]
    odesys, u0, tspan, p = return_ode_problem_default_inputs(egf)
    op = remake(op, u0=u0)
    predicted = DifferentialEquations.solve(op, solver, abstol=abstol, reltol=reltol, saveat=save_at);
    #Early exit if simulation could not be computed successfully.
    if predicted.retcode !== ReturnCode.Success
        Turing.@addlogprob! -Inf
        return
    end
    experimental_quantities = calculate_all_quantities(predicted)
    species_index = return_index_order_of_data_for_likelihood(egf)
    all_predictions = Array{Float64}(undef,0) #make sure type works as input for MvNormal
    [append!(all_predictions, experimental_quantities[i]) for i in species_index];

    #response to 2 EGF
    egf = dosages[2]
    odesys, u0, tspan, p = return_ode_problem_default_inputs(egf)
    op = remake(op, u0=u0)
    predicted = DifferentialEquations.solve(op, solver, abstol=abstol, reltol=reltol, saveat=save_at);
    #Early exit if simulation could not be computed successfully.
    if predicted.retcode !== ReturnCode.Success
        Turing.@addlogprob! -Inf
        return
    end
    experimental_quantities = calculate_all_quantities(predicted)
    species_index = return_index_order_of_data_for_likelihood(egf)
    [append!(all_predictions, experimental_quantities[i]) for i in species_index];

    #response to 0.2 EGF
    egf = dosages[3]
    odesys, u0, tspan, p = return_ode_problem_default_inputs(egf)
    op = remake(op, u0=u0)
    predicted = DifferentialEquations.solve(op, solver, abstol=abstol, reltol=reltol, saveat=save_at);
    #Early exit if simulation could not be computed successfully.
    if predicted.retcode !== ReturnCode.Success
        Turing.@addlogprob! -Inf
        return
    end
    experimental_quantities = calculate_all_quantities(predicted)
    species_index = return_index_order_of_data_for_likelihood(egf)
    [append!(all_predictions, experimental_quantities[i]) for i in species_index];

    #data likelihood
    data ~ MvNormal(all_predictions, σ) # MvNormal can take a vector input for standard deviation, and assumes covariance matrix is diagonal
end
