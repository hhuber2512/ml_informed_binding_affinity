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