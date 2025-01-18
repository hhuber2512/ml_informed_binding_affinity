using SBMLToolkit, ModelingToolkit

"""

egfr\\_ode(sbml\\_file: String,sbml\\_level: Int64, sbml\\_version: Int64)

Function loads an SBML.Model. \n 
ReactionSystem will need to be converted to a ModelingToolkit type for simulation. \n
May use Catalyst methods species() and parameters() \n

Should return: \n
Catalyst.ReactionSystem

"""
function egfr_ode(sbml_file, sbml_level, sbml_version)
    sbml_model = readSBML(sbml_file, doc -> begin
    set_level_and_version(sbml_level, sbml_version)(doc)
    convert_promotelocals_expandfuns(doc)
    end)
    rn = ReactionSystem(sbml_model)
    return rn
end