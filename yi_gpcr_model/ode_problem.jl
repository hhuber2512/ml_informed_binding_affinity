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