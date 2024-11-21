using ModelingToolkit

"""

calculate\\_active\\_G\\_protein\\_fraction(predicted:SciMLBase\\.ODESolution)

Calculate the fraction of active G protein \n 
Note - total G is defined as in Yi et al., that is as Gbg = G_total - G \n

Parameters: \n
predicted:SciMLBase\\.ODESolution: simulation used to calculate fraction of active G protein

Should return: \n
fraction_activated: Vector{Float64} of length n = number of time points at which measurements taken

"""
function calculate_active_G_protein_fraction(predicted)
    save_at = predicted.t
    #note - total G is defined in Yi et al. as Gbg = G_total - G
    total_G = [sum(Base.stack(predicted[[:Gbg, :G]])[:,i]) for i in 1:length(save_at)]
    fraction_activated = predicted[:Ga]./total_G;
    return fraction_activated
end

function simulate_dose_response_G(solver_inputs, op, ligand_dose)
    new_op = remake(op, u0=[:R=>10000.0, :L=>ligand_dose, :RL=>0.0, :Gd=>0, :Gbg=>3000.0, :G=>7000.0, :Ga=>0.0])
    predicted = DifferentialEquations.solve(new_op, solver_inputs["solver"], abstol=solver_inputs["abstol"], reltol=solver_inputs["reltol"], saveat=solver_inputs["saveat"])
    return calculate_active_G_protein_fraction(predicted)[1]
end

function simulate_dose_response(solver_inputs, op, ligand_dose)
    new_op = remake(op, u0=[:R=>10000.0, :L=>ligand_dose, :RL=>0.0, :Gd=>0, :Gbg=>3000.0, :G=>7000.0, :Ga=>0.0])
    predicted = DifferentialEquations.solve(new_op, solver_inputs["solver"], abstol=solver_inputs["abstol"], reltol=solver_inputs["reltol"], saveat=solver_inputs["saveat"])
    return predicted
end

function simulate_dose_response(solver_inputs, op, ligand_dose)
    new_op = remake(op, u0=[:R=>10000.0, :L=>ligand_dose, :RL=>0.0, :Gd=>0, :Gbg=>3000.0, :G=>7000.0, :Ga=>0.0])
    predicted = DifferentialEquations.solve(new_op, solver_inputs["solver"], abstol=solver_inputs["abstol"], reltol=solver_inputs["reltol"], saveat=solver_inputs["saveat"])
    return predicted
end

function simulate_dose_response_optimized(solver_inputs, op, ligand_dose, dose_setter)
    dose_setter(op, ligand_dose)
    predicted = DifferentialEquations.solve(op, solver_inputs["solver"], abstol=solver_inputs["abstol"], reltol=solver_inputs["reltol"], saveat=solver_inputs["saveat"])
    return predicted
end