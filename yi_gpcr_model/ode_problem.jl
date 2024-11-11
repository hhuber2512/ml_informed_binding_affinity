using Catalyst, DifferentialEquations
include("model.jl")

"""

return\\_ode\\_problem\\_default\\_inputs()

Function returns four inputs needed to define an ODEProblem from the DifferentialEquations package for GPCR signaling. \n
Species units are molecules, rate constants are 1/sec or 1/sec*molecules \n
Default initial conditions and parameter values were reported in Yi et al. \n

Should return: \n
odesys: ModelingToolkit.ODESystem \n
u0: Vector{Pair{Num, Float64}} 
tspan: Tuple{Int64, Int64} \n
p: Vector{Pair{Num, Float64}} 

"""
function return_ode_problem_default_inputs()
    reaction_network = gpcr_ode()
    #convert Catalyst.ReactionSystem to an ModelingToolkit.ODESystem
    odesys = complete(Catalyst.convert(ODESystem, reaction_network))
    tspan = (0,600)
    #define symbolic mappings for parameter (p) and initial condition (u0) inputs
    #this is the recommended approach per SciML https://docs.sciml.ai/ModelingToolkit/stable/basics/FAQ/#Transforming-value-maps-to-arrays
    p = [k_1=>3.32e-18, k_1inv=>0.01, k_2=>1.0, k_3=>1.0E-5, k_4=>4.0, k_5=>4.0E-4, k_6=>0.0040, k_7=>0.11]
    u0 = [R=>10000.0, L=>6.022E17, RL=>0.0, Gd=>3000.0, Gbg=>3000.0, G=>7000.0, Ga=>0.0]
    return odesys, u0, tspan, p
end

"""

return\\_ode\\_problem\\_solver\\_default\\_inputs()

Function returns four inputs needed to solve an ODEProblem from the DifferentialEquations package for GPCR signaling. \n
Species units are molecules, rate constants are 1/sec or 1/sec*molecules \n
Default initial conditions and parameter values were reported in Yi et al. \n

Should return a dictionary with the following entries: \n
solver: ModelingToolkit.ODESystem \n
abstol: Vector{Pair{Num, Float64}} 
reltol: Tuple{Int64, Int64} \n
saveat: Vector{Pair{Num, Float64}} 

"""
function return_ode_problem_solver_default_inputs(experimental_output)
    #absolute solver tolerance based on protein concentration deemed insignificant, 1e-5 nM, or less than 1 protein per cell
    abstol=1e-5
    #relative solver tolerance based on number of significant digits for protein concentration (5), plus 1: 1e-(5+1) 
    reltol=1e-6
    #stiff solver
    solver = QNDF()
    if experimental_output == "timecourse"
        saveat = deserialize("outputs/000_processed_active_G_timecourse.dict")["save_at"]
    elseif experimental_output == "dose_response"
        saveat = deserialize("outputs/000_processed_active_G_dose_response.dict")["save_at"]
    else 
        println("experiment for returning ode problem solver defaults not recognized, check input to return_ode_problem_solver_default_inputs")
    end
    return Dict("solver"=>solver, "abstol" => abstol, "reltol" => reltol, "saveat" => saveat)
end