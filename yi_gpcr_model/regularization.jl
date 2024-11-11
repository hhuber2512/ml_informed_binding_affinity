using Distributions

function regularize(regularization, k_1inv, k1)
    return Distributions.logpdf(Normal(log10(regularization["mean"]), regularization["std_dev"]), k_1inv - k_1)*regularization["lambda"] 
end