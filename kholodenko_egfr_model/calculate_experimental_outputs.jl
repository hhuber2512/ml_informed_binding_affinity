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
function calculate_all_quantities(sol)
    n = length(sol.t)
    #defined in Kholodenko
    total_phosphorylated_EGFR = sol[[:RGS, :RP, :RPLCgP, :RG, :RShP, :RShGS, :RShG, :RSh, :RPLCg]]
    total_phosphorylated_EGFR = [2*sum(mapreduce(permutedims, vcat, total_phosphorylated_EGFR)[i,:]) for i in 1:n]
    total_EGFR = sol[[:Ra, :RGS, :RP, :RPLCgP, :RG, :RShP, :RShGS, :RShG, :RSh, :RPLCg, :R, :R2]]
    total_EGFR = [2*sum(mapreduce(permutedims, vcat, total_EGFR)[i,:]) for i in 1:n]
    percent_phosphorylated_EGFR = total_phosphorylated_EGFR./total_EGFR
    
    total_phosphorylated_SHC = sol[[:ShGS, :RShP, :RShGS, :RShG, :ShP, :ShG]]
    total_phosphorylated_SHC = [sum(mapreduce(permutedims, vcat, total_phosphorylated_SHC)[i,:]) for i in 1:n]
    total_SHC = sol[[:Shc, :ShGS, :RShP, :RShGS, :RShG, :ShP, :RSh, :ShG]]
    total_SHC = [sum(mapreduce(permutedims, vcat, total_SHC)[i,:]) for i in 1:n]
    percent_phosphorylated_SHC = total_phosphorylated_SHC./total_SHC

    total_phosphorylated_PLCg = sol[[:RPLCgP, :PLCgP]]
    total_phosphorylated_PLCg = [sum(mapreduce(permutedims, vcat, total_phosphorylated_PLCg)[i,:]) for i in 1:n]
    total_PLCg = sol[[:RPLCgP, :PLCg, :PLCgl, :RPLCg, :PLCgP]]
    total_PLCg = [sum(mapreduce(permutedims, vcat, total_PLCg)[i,:]) for i in 1:n]
    percent_phosphorylated_PLCg = total_phosphorylated_PLCg./total_PLCg
    
    GRB2_EGFR = sol[[:RGS, :RG, :RShGS, :RShG]]
    GRB2_EGFR = [sum(mapreduce(permutedims, vcat, GRB2_EGFR)[i,:]) for i in 1:n]
    percent_GRB2_EGFR = GRB2_EGFR./total_EGFR

    GRB2_SHC = sol[[:ShGS, :RShGS, :RShG, :ShG]]
    GRB2_SHC = [sum(mapreduce(permutedims, vcat, GRB2_SHC)[i,:]) for i in 1:n]
    percent_GRB2_SHC = GRB2_SHC./total_SHC

    return Dict("p_egfr" => 100*percent_phosphorylated_EGFR, "p_shc" => 100*percent_phosphorylated_SHC, 
    "p_plcg" => 100*percent_phosphorylated_PLCg, "grb2_egfr" => 100*percent_GRB2_EGFR, "grb2_shc" => 100*percent_GRB2_SHC)

end
