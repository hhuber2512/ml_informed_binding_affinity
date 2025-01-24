using Catalyst, DifferentialEquations
include("model.jl")

"""

return\\_ode\\_problem\\_default\\_inputs() \n

Function returns four inputs needed to define an ODEProblem from the DifferentialEquations package for EGFR signaling. \n
Species units are molecules, rate constants are 1/sec or 1/sec*nM \n
Default initial conditions and parameter values were reported in Kholodenko et al. \n

Should return: \n
odesys: ModelingToolkit.ODESystem \n
u0: Vector{Pair{Num, Float64}} \n
tspan: Tuple{Int64, Int64} \n
p: Vector{Pair{Num, Float64}} \n

"""
function return_ode_problem_default_inputs(EGF_concentration)
    reaction_network = egfr_ode("data/kholodenko1.xml",3,1);
    #convert Catalyst.ReactionSystem to an ModelingToolkit.ODESystem
    odesys = complete(Catalyst.convert(ODESystem, reaction_network));
    tspan = (0,120)
    t_expt = [0, 15, 30, 45, 60, 120]
    prob = ODEProblem(odesys, [], tspan, [], saveat=t_expt)
    #define symbolic mappings for parameter (p) and initial condition (u0) inputs
    #this is the recommended approach per SciML https://docs.sciml.ai/ModelingToolkit/stable/basics/FAQ/#Transforming-value-maps-to-arrays
    p = [:k7b => 0.006, :k22f => 0.03, :k23b => 0.021, :k22b => 0.064, :k20f => 0.12, :k24b => 0.0429, :k14f => 6.0, :k25b => 0.03, :k5b => 0.2, :k13b => 0.6, 
    :k23f => 0.1, :k7f => 0.3, :k21f => 0.003, :k11b => 0.0045, :k21b => 0.1, :V4 => 450.0, :k1f => 0.003, :k18b => 0.0009, :k12f => 0.0015, :k25f => 1.0, :k19f => 0.01, 
    :k24f => 0.009, :k6f => 1.0, :k3f => 1.0, :k17f => 0.003, :k9b => 0.05, :K16 => 340.0, :V8 => 1.0, :k11f => 0.03, :k14b => 0.06, :k6b => 0.05, :k5f => 0.06, :k15b => 0.0009, 
    :k19b => 0.0214, :k13f => 0.09, :k1b => 0.06, :k12b => 0.0001, :K4 => 50.0, :K8 => 100.0, :k17b => 0.1, :ADP => 1.0, :k3b => 0.01, :k9f => 0.003, :ATP => 1.0, :k20b => 0.00024, 
    :k10f => 0.01, :V16 => 1.7, :k15f => 0.3, :k2b => 0.1, :k18f => 0.3, :k2f => 0.01, :k10b => 0.06, :default_compartment => 1.0] 
    #to get initial conditions, we extract values included in the SBML file and map them
    #quick note on initial conditions - if you look at EGF's initial concentration, it is 680. This corresponds to an input of 20 nM, 
    #if we consider the rescaling that was done in the paper - ie. divide by 33.33. See details in Table II of Kholodenko
    if EGF_concentration == 20
        u0 = species(reaction_network) .=> prob.u0
        u0 = [:Ra => 0.0, :RGS => 0.0, :Shc => 150.0, :EGF => 680.0, :ShGS => 0.0, :RP => 0.0, :RPLCgP => 0.0, :RG => 0.0, :RShP => 0.0, :SOS => 34.0, 
        :RShGS => 0.0, :RShG => 0.0, :PLCg => 105.0, :GS => 0.0, :ShP => 0.0, :Grb => 85.0, :PLCgl => 0.0, :RSh => 0.0, :ShG => 0.0, :RPLCg => 0.0, 
        :PLCgP => 0.0, :R => 100.0, :R2 => 0.0]
    elseif EGF_concentration == 2
        u0 = [:Ra => 0.0, :RGS => 0.0, :Shc => 150.0, :EGF => 68.0, :ShGS => 0.0, :RP => 0.0, :RPLCgP => 0.0, :RG => 0.0, :RShP => 0.0, :SOS => 34.0, 
        :RShGS => 0.0, :RShG => 0.0, :PLCg => 105.0, :GS => 0.0, :ShP => 0.0, :Grb => 85.0, :PLCgl => 0.0, :RSh => 0.0, :ShG => 0.0, :RPLCg => 0.0, 
        :PLCgP => 0.0, :R => 100.0, :R2 => 0.0]
    elseif EGF_concentration == 0.2
        u0 = [:Ra => 0.0, :RGS => 0.0, :Shc => 150.0, :EGF => 6.80, :ShGS => 0.0, :RP => 0.0, :RPLCgP => 0.0, :RG => 0.0, :RShP => 0.0, :SOS => 34.0, 
        :RShGS => 0.0, :RShG => 0.0, :PLCg => 105.0, :GS => 0.0, :ShP => 0.0, :Grb => 85.0, :PLCgl => 0.0, :RSh => 0.0, :ShG => 0.0, :RPLCg => 0.0, 
        :PLCgP => 0.0, :R => 100.0, :R2 => 0.0]
    else
        print("check EGF concentration input")
    end
    return odesys, u0, tspan, p
end

"""

return\\_ode\\_problem\\_solver\\_default\\_inputs()

Function returns four inputs needed to solve an ODEProblem from the DifferentialEquations package for EGFR signaling. \n
Species units are molecules, rate constants are 1/sec or 1/sec*nM \n
Default initial conditions and parameter values were reported in Kholodenko et al. \n

Should return: Dict with following keys: \n
solver: ModelingToolkit.ODESystem \n
abstol: Vector{Pair{Num, Float64}} \n
reltol: Tuple{Int64, Int64} \n
saveat: Vector{Pair{Num, Float64}} \n

"""
function return_ode_problem_solver_default_inputs()
    #absolute solver tolerance based on protein concentration deemed insignificant, 1e-5 nM, or less than 1 protein per cell
    abstol=1e-5
    #relative solver tolerance based on number of significant digits for protein concentration (5), plus 1: 1e-(5+1) 
    reltol=1e-6
    #stiff solver
    solver = QNDF()
    #all species observed at the same timepoints, can take any save_at
    saveat = deserialize("outputs/000_processed_grb_egfr_20.dict")["save_at"]
    return Dict("solver"=>solver, "abstol" => abstol, "reltol" => reltol, "saveat" => saveat)
end