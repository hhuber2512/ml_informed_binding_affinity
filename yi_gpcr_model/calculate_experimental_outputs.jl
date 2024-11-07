using ModelingToolkit

#declare variables in Main using ModelingToolkit macro, s.t. symbolic indexing may be used
#declare it here, rather than in the function, as we don't want to call the macro everytime the function is run
@variables t L(t) R(t) RL(t) Gd(t) Gbg(t) G(t) Ga(t) 
@parameters k_1 k_1inv k_2 k_3 k_4 k_5 k_6 k_7

"""

calculate\\_active\\_G\\_protein\\_fraction(predicted:SciMLBase.ODESolution)

Function calculates the fraction of active G protein \n 
Note - total G is defined in Yi et al. as Gbg = G_total - G \n

Should return: \n
fraction_activated: Vector{Float64} of length n = number of time points at which measurements taken

"""

function calculate_active_G_protein_fraction(predicted)
    save_at = predicted.t
    #note - total G is defined in Yi et al. as Gbg = G_total - G
    total_G = [sum(Base.stack(predicted[[Gbg, G]])[:,i]) for i in 1:length(save_at)]
    fraction_activated = predicted[Ga]./total_G;
    return fraction_activated
end