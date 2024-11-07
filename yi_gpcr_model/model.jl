using Catalyst, ModelingToolkit

"""

gpcr\\_ode()

Function defines a (complete) Catalyst.ReactionSystem via via rules-based modeling macro @reaction\\_network. \n 
ReactionSystem will need to be converted to a ModelingToolkit type for simulation. \n
May use Catalyst methods species() and parameters() \n
Species units are molecules, rate constants are 1/sec or 1/sec*molecules \n
We defined default initial conditions and parameter values, which were reported in Yi et al. \n

Should return: \n
Catalyst.ReactionSystem

"""
function gpcr_ode()
    rn = Catalyst.@reaction_network begin
        @parameters k_1=3.32e-18 k_1inv=0.01 k_2=1.0 k_3=1.0E-5 k_4=4.0 k_5=4.0E-4 k_6=0.0040 k_7=0.11
        @species R(t)=10000.0 L(t)=6.022E17 RL(t)=0.0 Gd(t)=3000.0 Gbg(t)=3000.0 G(t)=7000.0 Ga(t)=0.0
        (k_1,k_1inv), L + R <--> RL
        k_2, Gd + Gbg --> G
        k_3, RL + G --> Ga + Gbg + RL
        (k_4, k_5), 0 <--> R
        k_6, RL --> 0
        k_7, Ga --> Gd
    end
    return rn
end