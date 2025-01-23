using Distributions

"""

regularize(regularization:Dict{String, Real}, k_1inv:Float64, k_1:Float64, parameter_name:String) \n

This function returns the log probability of k_1inv - k_1, with respect to a normal distribution. \n
The normal distribution is a function of the mean and standard deviation, given as inputs to the function. \n
Note: distribution is being evaluated on log-transformed variables and mean. This is why we evaluate the probability of the difference between k_1inv and k_1, rather than the ratio\n
That is, the log(x/y) = log(x)-log(y) \n
Finally, we multiply by hyperparameter lambda, which represents the confidence of the mean. \n
Here, lamda = alphafold ranking score (0-1) of protein structure prediction \n

Parameters: \n
regularization: Dict{String, Real} \n
Dictionary with keys "mean", "std_dev", and "lambda", which are the hyperparameters needed to define the regularization \n
IMPORTANT: mean should be on log10 scale when inputted \n
k_1inv:Float64 \n
Proposed value for k_1inv \n
k_1:Float64 \n
Proposed value for k_1 \n

Should return: \n
log probability, type: Float64
"""

function regularize(regularization, k_1inv, k_1, parameter_name)
    return Distributions.logpdf(Normal(regularization[parameter_name]["mean"], regularization[parameter_name]["std_dev"]), k_1inv - k_1)*regularization[parameter_name]["lambda"]  
end