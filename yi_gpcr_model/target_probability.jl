using Turing, Distributions, Catalyst, DifferentialEquations
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

Turing.@model function logprob_regularized(data, odeproblem, σ, regularization, odesolver_timecourse, odesolver_dose_response, ligand_dose, normalization_dose)
    #define prior distributions
    k_1 ~ Uniform(-20,-16) #k_1 = 3.32e-18
    k_1inv ~ Uniform(-4,0) #kinv = 1e-2
    k_2 ~ Uniform(-2,2) #k_2=>1.0 (1E0)
    k_3 ~ Uniform(-7,-3) #k_3=>1.0E-5 
    k_4 ~ Uniform(-2,2) #k_4=>4.0 (4E0)
    k_5 ~ Uniform(-6,-2) #k_5=>4.0E-4
    k_6 ~ Uniform(-5,-1) #k_6=>0.0040 (4E-3)
    k_7 ~ Uniform(-3,1) #k_7=>0.11 (1.1E-1)

    #regularization term, logscale 
    Turing.@addlogprob! regularize(regularization, k_1inv, k_1)

    #initial protein concentration units are molecules (here, we only infer one initial condition)
    G ~ Normal(log(7000), 1) #prior initialized about ground-truth initial condition, 7000 molecules
    
    p_sampled = [10.0^k_1, 10.0^k_1inv, 10.0^k_2, 10.0^k_3, 10.0^k_4, 10.0^k_5, 10.0^k_6, 10.0^k_7]
    
    #redefine ode problem with sampled values
    #per SciML docs, we make sure to input parameter map, rather than rely on parameter order
    p = [:k_1 => p_sampled[1], :k_1inv => p_sampled[2], :k_2 => p_sampled[3], :k_3 => p_sampled[4], :k_4 => p_sampled[5], 
    :k_5 => p_sampled[6], :k_6 => p_sampled[7], :k_7 => p_sampled[8]]
    u0 = [:R => 10000.0, :L => 6.022E17, :RL => 0.0, :Gd => 3000.0, :Gbg => 3000.0, :G => exp(G), :Ga => 0.0]  
    op = remake(odeproblem, u0=u0, p=p)

    #timecourse simulation
    save_at = odesolver_timecourse["saveat"]
    solver = odesolver_timecourse["solver"]
    abstol = odesolver_timecourse["abstol"]
    reltol = odesolver_timecourse["reltol"]
    predicted = DifferentialEquations.solve(op, solver, abstol=abstol, reltol=reltol, saveat=save_at);
    #Early exit if simulation could not be computed successfully.
    if predicted.retcode !== ReturnCode.Success
        Turing.@addlogprob! -Inf
        return
    end
    fraction_activated_timecourse = calculate_active_G_protein_fraction(predicted)

    #dose response simulation
    #need only redefine save_at, all other solver inputs are the same
    save_at = odesolver_dose_response["saveat"]
    
    predicted_responses = []
    for i in ligand_dose
        u0 = [:R => 10000.0, :L => i, :RL => 0.0, :Gd => 3000.0, :Gbg => 3000.0, :G => exp(G), :Ga => 0.0] 
        op = remake(op,u0=u0)
        predicted = DifferentialEquations.solve(op, QNDF(), abstol=1e-5, reltol=1e-6, saveat=save_at);
        #Early exit if simulation could not be computed successfully.
        if predicted.retcode !== ReturnCode.Success
            Turing.@addlogprob! -Inf
            return
        end
        fraction_activated = calculate_active_G_protein_fraction(predicted)
        append!(predicted_responses, fraction_activated)
    end

    #extract response we normalize to
    u0 = [:R => 10000.0, :L => normalization_dose, :RL => 0.0, :Gd => 3000.0, :Gbg => 3000.0, :G => exp(G), :Ga => 0.0]
    op = remake(op, u0=u0)
    predicted = DifferentialEquations.solve(op, QNDF(), abstol=1e-5, reltol=1e-6, saveat=save_at);
    normalize_to = calculate_active_G_protein_fraction(predicted)

    #note, 1 uM corresponds to last stored dose response
    scaled_response = predicted_responses./normalize_to;

    #combine all predictions into an array
    all_predictions = vcat(fraction_activated_timecourse, scaled_response)

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
Turing.@model function logprob_unregularized(data, odeproblem, σ, odesolver_timecourse, odesolver_dose_response, ligand_dose, normalization_dose)
    #define prior distributions
    k_1 ~ Uniform(-20,-16) #k_1 = 3.32e-18
    k_1inv ~ Uniform(-4,0) #kinv = 1e-2
    k_2 ~ Uniform(-2,2) #k_2=>1.0 (1E0)
    k_3 ~ Uniform(-7,-3) #k_3=>1.0E-5 
    k_4 ~ Uniform(-2,2) #k_4=>4.0 (4E0)
    k_5 ~ Uniform(-6,-2) #k_5=>4.0E-4
    k_6 ~ Uniform(-5,-1) #k_6=>0.0040 (4E-3)
    k_7 ~ Uniform(-3,1) #k_7=>0.11 (1.1E-1)

    #initial protein concentration units are molecules (here, we only infer one initial condition)
    G ~ Normal(log(7000), 1) #prior initialized about ground-truth initial condition, 7000 molecules
    
    p_sampled = [10.0^k_1, 10.0^k_1inv, 10.0^k_2, 10.0^k_3, 10.0^k_4, 10.0^k_5, 10.0^k_6, 10.0^k_7]
    
    #redefine ode problem with sampled values
    #per SciML docs, we make sure to input parameter map, rather than rely on parameter order
    p = [:k_1 => p_sampled[1], :k_1inv => p_sampled[2], :k_2 => p_sampled[3], :k_3 => p_sampled[4], :k_4 => p_sampled[5], 
    :k_5 => p_sampled[6], :k_6 => p_sampled[7], :k_7 => p_sampled[8]]
    u0 = [:R => 10000.0, :L => 6.022E17, :RL => 0.0, :Gd => 3000.0, :Gbg => 3000.0, :G => exp(G), :Ga => 0.0]  
    op = remake(odeproblem, u0=u0, p=p)

    #timecourse simulation
    save_at = odesolver_timecourse["saveat"]
    solver = odesolver_timecourse["solver"]
    abstol = odesolver_timecourse["abstol"]
    reltol = odesolver_timecourse["reltol"]
    predicted = DifferentialEquations.solve(op, solver, abstol=abstol, reltol=reltol, saveat=save_at);
    #Early exit if simulation could not be computed successfully.
    if predicted.retcode !== ReturnCode.Success
        Turing.@addlogprob! -Inf
        return
    end
    fraction_activated_timecourse = calculate_active_G_protein_fraction(predicted)

    #dose response simulation
    #need only redefine save_at, all other solver inputs are the same
    save_at = odesolver_dose_response["saveat"]
    
    predicted_responses = []
    for i in ligand_dose
        u0 = [:R => 10000.0, :L => i, :RL => 0.0, :Gd => 3000.0, :Gbg => 3000.0, :G => exp(G), :Ga => 0.0] 
        op = remake(op,u0=u0)
        predicted = DifferentialEquations.solve(op, QNDF(), abstol=1e-5, reltol=1e-6, saveat=save_at);
        #Early exit if simulation could not be computed successfully.
        if predicted.retcode !== ReturnCode.Success
            Turing.@addlogprob! -Inf
            return
        end
        fraction_activated = calculate_active_G_protein_fraction(predicted)
        append!(predicted_responses, fraction_activated)
    end

    #extract response we normalize to
    u0 = [:R => 10000.0, :L => normalization_dose, :RL => 0.0, :Gd => 3000.0, :Gbg => 3000.0, :G => exp(G), :Ga => 0.0]
    op = remake(op, u0=u0)
    predicted = DifferentialEquations.solve(op, QNDF(), abstol=1e-5, reltol=1e-6, saveat=save_at);
    normalize_to = calculate_active_G_protein_fraction(predicted)

    #note, 1 uM corresponds to last stored dose response
    scaled_response = predicted_responses./normalize_to;

    #combine all predictions into an array
    all_predictions = vcat(fraction_activated_timecourse, scaled_response)

    #data likelihood
    data ~ MvNormal(all_predictions, σ) # MvNormal can take a vector input for standard deviation, and assumes covariance matrix is diagonal
end
