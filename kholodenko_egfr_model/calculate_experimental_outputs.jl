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

function return_index_order_of_data_for_likelihood(egf_dosage)
    if egf_dosage == 20
        return ["p_egfr", "p_plcg", "grb2_egfr", "grb2_shc"]
    elseif egf_dosage == 2
        return ["p_egfr", "p_plcg"]
    elseif egf_dosage == 0.2
        return ["p_egfr"]
    end
end

function return_ligand_dose_order_for_likelihood()
    return [20, 2, 0.2]
end

function return_training_data_names()
    return ["000_processed_p_egfr_20.dict", "000_processed_p_plcg_20.dict", "000_processed_grb_egfr_20.dict", "000_processed_grb_shc_20.dict",
    "000_processed_p_egfr_2.dict", "000_processed_p_plcg_2.dict", "000_processed_p_egfr_02.dict"]
end

function extract_likelihood_species(experimental_quantities, egf_dosage)
    quantities_per_dose = Array{Float64}(undef,0)
    species_index = return_index_order_of_data_for_likelihood(egf_dosage)
    [append!(quantities_per_dose, experimental_quantities[j]) for j in species_index]
    return quantities_per_dose
end
