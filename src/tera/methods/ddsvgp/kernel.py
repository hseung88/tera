import torch
from gpytorch.kernels.rbf_kernel import RBFKernel as rbf_kernel_type, postprocess_rbf


class rbf_kernel_directional_grad(rbf_kernel_type):
    def forward(self, x1, x2, diag=False, **params):
        batch_shape = x1.shape[:-2]
        n_batch_dims = len(batch_shape)
        n1, d = x1.shape[-2:]
        n2 = x2.shape[-2]

        v1 = params["v1"]
        v2 = params["v2"]

        n_dir1 = int(v1.shape[-2] / n1)
        n_dir2 = int(v2.shape[-2] / n2)
        assert n_dir1 == n_dir2, "v1 and v2 must contain same number of directions"

        self.set_num_directions(n_dir1)

        v1 = (v1.T / torch.norm(v1, dim=1)).T
        v2 = (v2.T / torch.norm(v2, dim=1)).T

        if not diag:
            covariance = torch.zeros(
                *batch_shape, n1 * (n_dir1 + 1), n2 * (n_dir2 + 1), device=x1.device, dtype=x1.dtype
            )
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)

            diff = self.covar_dist(x1_, x2_, square_dist=True, **params)
            diff = postprocess_rbf(diff)
            k_11 = diff
            covariance[..., :n1, :n2] = k_11

            x2_v2 = x2_.reshape(n2, 1, d).bmm(torch.transpose(v2.reshape(n2, n_dir2, d), -2, -1))
            x1_v2 = x1_ @ v2.T
            outer = x1_v2 - x2_v2.flatten()

            pi1 = torch.arange(n2 * (n_dir2)).view(n2, n_dir2).t().reshape((n2 * (n_dir2)))

            outer1 = outer[:, pi1] / self.lengthscale.unsqueeze(-2)

            covariance[..., :n1, n2:] = outer1 * k_11.repeat([*([1] * (n_batch_dims + 1)), n_dir2])

            x1_v1 = x1_.reshape(n1, 1, d).bmm(torch.transpose(v1.reshape(n1, n_dir1, d), -2, -1))
            x2_v1 = x2_ @ v1.T
            outer = x1_v1.flatten() - x2_v1

            pi2 = torch.arange(n1 * (n_dir1)).view(n1, n_dir1).t().reshape((n1 * (n_dir1)))
            outer2 = outer[:, pi2]
            outer2 = outer2.t() / self.lengthscale.unsqueeze(-2)
            covariance[..., n1:, :n2] = -outer2 * k_11.repeat([n_dir1, *([1] * (n_batch_dims + 1))])

            outer3 = outer1.repeat(1, n_dir2, 1) * outer2.repeat(1, 1, n_dir1)

            kp = v1 @ v2.T / self.lengthscale.pow(2)
            kp = kp[:, pi1][pi2, :]
            chain_rule = kp - outer3
            covariance[..., n1:, n2:] = chain_rule * k_11.repeat([*([1] * n_batch_dims), n_dir1, n_dir2])

            pi1 = (
                torch.arange(n1 * (n_dir1 + 1))
                .view(n_dir1 + 1, n1)
                .t()
                .reshape((n1 * (n_dir1 + 1)))
            )
            pi2 = (
                torch.arange(n2 * (n_dir2 + 1))
                .view(n_dir2 + 1, n2)
                .t()
                .reshape((n2 * (n_dir2 + 1)))
            )
            covariance = covariance[..., pi1, :][..., :, pi2]
            return covariance

        else:
            if not (
                n1 == n2 and torch.eq(x1, x2).all() and n_dir1 == n_dir2 and torch.eq(v1, v2).all()
            ):
                raise RuntimeError("diag=True only works when x1 == x2 and v1 == v2")

            kernel_diag = super(rbf_kernel_directional_grad, self).forward(x1, x2, diag=True)
            grad_diag = torch.ones(
                *batch_shape, n2, n_dir2, device=x1.device, dtype=x1.dtype
            ) / self.lengthscale.pow(2)
            grad_diag = grad_diag.transpose(-1, -2).reshape(*batch_shape, n2 * n_dir2)
            k_diag = torch.cat((kernel_diag, grad_diag), dim=-1)
            pi = (
                torch.arange(n2 * (n_dir2 + 1))
                .view(n_dir2 + 1, n2)
                .t()
                .reshape((n2 * (n_dir2 + 1)))
            )
            return k_diag[..., pi]

    def set_num_directions(self, num_directions):
        self.n_dir1 = num_directions

    def num_outputs_per_input(self, x1, x2):
        return self.n_dir1 + 1
