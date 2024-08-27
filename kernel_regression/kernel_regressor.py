import torch
from kernels import gaussian_kernel


class KernelRegressionLayer(torch.nn.Module):
    def __init__(self, X_train, y_train, kernel=None, **kwargs):
        """
        A layer that will perform kernel regression using X_train, y_train
        and a custom kernel function. For use in predicting time
        dependent data, order X_train and y_train's rows by time
        with lower rows representing future points.

        Parameters
        ----------
        X_train (tensor): (n x d) (Possibly transformed) tensor of samples to
          be used as neighbors in prediction

        y_train (tensor): (n x 1) neighbors' target values

        kernel (function, optional): Kernel to be used.
          Default is a Gaussian kernel w/ bandwidth one.
        """
        super().__init__()

        self.X_train = X_train
        self.y_train = y_train

        if kernel is None:
            self.kernel = gaussian_kernel
        else:
            self.kernel = kernel

    def forward(self, x, p=2, hide_future=False,
                leave_one_out=False, **kwargs):
        """
        Predict the target value of a point x using kernel regression.

        Parameters
        ----------
        x (tensor): (m x d) (Possibly transformed) samples we want to predict

        p (float, optional): Can be negative. Degree of L^p norm for distance.

        hide_future (bool): Hide future points or not in prediction.
            Generally, we only set hide_future = True if x == X_train
            and there is time dependence

        leave_one_out (bool): Boolean that dictates whether or not
            we exclude a point from its own training set
            when predicting. ONLY SET THIS TO TRUE IF x == X_train

        **kwargs: Any additional parameters for our kernel (e.g. bandwidth)

        Output
        ----------
        predicted_y (tensor): (m x 1) Predictions of x's target value
        """
        squared_dists = torch.cdist(x, self.X_train, p=p)**2

        # If we wish to hide future points, then the distance of
        # a point from those that come after it must register as infinity.
        # That way, the kernel values for those points are set to 0
        if hide_future:
            infty_setter = torch.triu(float('inf') +
                                      torch.ones(squared_dists.shape),
                                      diagonal=1)
            squared_dists += infty_setter

        if leave_one_out:
            squared_dists.fill_diagonal_(float('inf'))

        kernel_weights = self.kernel(squared_dist=squared_dists, **kwargs)

        predicted_y = torch.matmul(kernel_weights, self.y_train)

        return predicted_y
