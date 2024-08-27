import torch


def gaussian_kernel(squared_dist, bandwidth=1):
    """
    Given a tensor of squared distances,
    outputs a tensor of Gaussian kernel values for use as weights
    in kernel regression.

    Arguments
    ----------
    squared_dist (Tensor): (n x m) tensor of squared distances

    bandwidth (float): bandwidth in Gaussian kernel output

    Output
    ----------
    kernel_values (Tensor): (n x m) tensor of Gaussian kernel values
    """
    scaled_squared_dist = squared_dist/bandwidth
    kernel_values = torch.exp(- scaled_squared_dist
                              - torch.logsumexp(- scaled_squared_dist,
                                                axis=1, keepdim=True))

    return kernel_values


def epanechnikov_kernel(squared_dist, bandwidth=1):
    """
    Given a tensor of squared distances,
    outputs a tensor of Epanechnikov kernel values for use as weights
    in kernel regression.

    Arguments
    ----------
    squared_dist (Tensor): (n x m) tensor of squared distances

    bandwidth (float): bandwidth in Epanechnikov kernel output

    Output
    ----------
    kernel_values (Tensor): (n x m) tensor of Epanechnikov kernel values
    """
    scaled_squared_dist = (squared_dist/bandwidth)
    scaled_squared_dist[torch.abs(scaled_squared_dist) > 1] = 1

    kernel_values = 0.75*(1-scaled_squared_dist)
    kernel_values = kernel_values/torch.sum(kernel_values,
                                            axis=1,
                                            keepdim=True)

    return kernel_values
