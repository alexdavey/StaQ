import torch as th
from functools import partial
from copy import deepcopy


class MetricGradPO:
    def __init__(self, policy, lr, cg_max_steps=120, max_step=None, verbose=False):
        self.cg_max_steps = cg_max_steps
        self.cg_damping = 0.01
        self.policy = policy
        self.lr = lr
        self.max_step = max_step
        self.verbose = verbose


    def conjugate_gradient_solver(self,
            matrix_vector_dot_fn,
            b,
            max_iter=10,
            residual_tol=1e-10,
    ) -> th.Tensor:
        """
        Finds an approximate solution to a set of linear equations Ax = b

        Sources:
         - https://github.com/ajlangley/trpo-pytorch/blob/master/conjugate_gradient.py
         - https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py#L122

        Reference:
         - https://epubs.siam.org/doi/abs/10.1137/1.9781611971446.ch6

        :param matrix_vector_dot_fn:
            a function that right multiplies a matrix A by a vector v
        :param b:
            the right hand term in the set of linear equations Ax = b
        :param max_iter:
            the maximum number of iterations (default is 10)
        :param residual_tol:
            residual tolerance for early stopping of the solving (default is 1e-10)
        :return x:
            the approximate solution to the system of equations defined by `matrix_vector_dot_fn`
            and b
        """

        # The vector is not initialized at 0 because of the instability issues when the gradient becomes small.
        # A small random gaussian noise is used for the initialization.
        x = 1e-4 * th.randn_like(b)
        residual = b - matrix_vector_dot_fn(x)
        # Equivalent to th.linalg.norm(residual) ** 2 (L2 norm squared)
        residual_squared_norm = th.matmul(residual, residual)

        if residual_squared_norm < residual_tol:
            # If the gradient becomes extremely small
            # The denominator in alpha will become zero
            # Leading to a division by zero
            return x

        p = residual.clone()

        for i in range(max_iter):
            # A @ p (matrix vector multiplication)
            A_dot_p = matrix_vector_dot_fn(p)

            alpha = residual_squared_norm / p.dot(A_dot_p)
            x += alpha * p

            if i == max_iter - 1:
                return x

            residual -= alpha * A_dot_p
            new_residual_squared_norm = th.matmul(residual, residual)

            if new_residual_squared_norm < residual_tol:
                return x

            beta = new_residual_squared_norm / residual_squared_norm
            residual_squared_norm = new_residual_squared_norm
            p = residual + beta * p
        # Note: this return statement is only used when max_iter=0
        return x

    def _compute_actor_grad(
        self, kl_div: th.Tensor, policy_objective: th.Tensor
    ) :
        """
        Compute actor gradients for kl div and surrogate objectives.

        :param kl_div: The KL divergence objective
        :param policy_objective: The surrogate objective ("classic" policy gradient)
        :return: List of actor params, gradients and gradients shape.
        """
        # This is necessary because not all the parameters in the policy have gradients w.r.t. the KL divergence
        # The policy objective is also called surrogate objective
        policy_objective_gradients_list = []
        # Contains the gradients of the KL divergence
        grad_kl_list = []
        # Contains the shape of the gradients of the KL divergence w.r.t each parameter
        # This way the flattened gradient can be reshaped back into the original shapes and applied to
        # the parameters
        grad_shape = []
        # Contains the parameters which have non-zeros KL divergence gradients
        # The list is used during the line-search to apply the step to each parameters
        actor_params = []

        for name, param in self.policy.named_parameters():
            # Skip parameters related to value function based on name
            # this work for built-in policies only (not custom ones)
            if "value" in name:
                continue

            # For each parameter we compute the gradient of the KL divergence w.r.t to that parameter
            kl_param_grad, *_ = th.autograd.grad(
                kl_div,
                param,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
                only_inputs=True,
            )
            # If the gradient is not zero (not None), we store the parameter in the actor_params list
            # and add the gradient and its shape to grad_kl and grad_shape respectively
            if kl_param_grad is not None:
                # If the parameter impacts the KL divergence (i.e. the policy)
                # we compute the gradient of the policy objective w.r.t to the parameter
                # this avoids computing the gradient if it's not going to be used in the conjugate gradient step
                policy_objective_grad, *_ = th.autograd.grad(policy_objective, param, retain_graph=True, only_inputs=True)

                grad_shape.append(kl_param_grad.shape)
                grad_kl_list.append(kl_param_grad.reshape(-1))
                policy_objective_gradients_list.append(policy_objective_grad.reshape(-1))
                actor_params.append(param)

        # Gradients are concatenated before the conjugate gradient step
        policy_objective_gradients = th.cat(policy_objective_gradients_list)
        grad_kl = th.cat(grad_kl_list)
        return actor_params, policy_objective_gradients, grad_kl, grad_shape

    def hessian_vector_product(
        self, params, grad_kl, vector, retain_graph = True
    ) :
        """
        Computes the matrix-vector product with the Fisher information matrix.

        :param params: list of parameters used to compute the Hessian
        :param grad_kl: flattened gradient of the KL divergence between the old and new policy
        :param vector: vector to compute the dot product the hessian-vector dot product with
        :param retain_graph: if True, the graph will be kept after computing the Hessian
        :return: Hessian-vector dot product (with damping)
        """
        jacobian_vector_product = (grad_kl * vector).sum()
        return self.flat_grad(jacobian_vector_product, params, retain_graph=retain_graph) + self.cg_damping * vector

    def flat_grad(self,
            output,
            parameters,
            create_graph: bool = False,
            retain_graph: bool = False,
    ) -> th.Tensor:
        """
        Returns the gradients of the passed sequence of parameters into a flat gradient.
        Order of parameters is preserved.

        :param output: functional output to compute the gradient for
        :param parameters: sequence of ``Parameter``
        :param retain_graph: If ``False``, the graph used to compute the grad will be freed.
            Defaults to the value of ``create_graph``.
        :param create_graph: If ``True``, graph of the derivative will be constructed,
            allowing to compute higher order derivative products. Default: ``False``.
        :return: Tensor containing the flattened gradients
        """
        grads = th.autograd.grad(
            output,
            parameters,
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        return th.cat([th.ravel(grad) for grad in grads if grad is not None])

    def step(self, objective_n_metric):
        self.zero_grad()

        objective_val, metric_val = objective_n_metric(self.policy)

        actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(metric_val, objective_val)

        # Hessian-vector dot product function used in the conjugate gradient step
        hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)

        # Computing search direction
        nat_g = self.conjugate_gradient_solver(
            hessian_vector_product_fn,
            policy_objective_gradients,
            max_iter=self.cg_max_steps,
        )
        if self.max_step is None:
            self.update_weights(nat_g, self.lr)
        else:
            self.update_weights_constrained(nat_g, self.max_step, lambda x: objective_n_metric(x, kl_to_old=True))


    def update_weights(self, flat_delta, lr):
        idx = 0
        for p in self.policy.parameters():
            numel = p.numel()
            p.data = p + lr * flat_delta[idx:idx + numel].view(p.shape)
            idx += numel

    def set_weights(self, weights):
        for p, w in zip(self.policy.parameters(), weights):
            p.data = w

    def zero_grad(self):
        [g.grad.zero_() for g in self.policy.parameters() if g.grad is not None]

    def update_weights_constrained(self, nat_g, max_distance, metric, nb_search=100):
        # line search variables
        if self.verbose:
            print('-------------starting line search')
        stepsize = self.lr
        for nb_eff_search in range(nb_search):
            pol = deepcopy(self.policy)
            self.update_weights(nat_g, stepsize)
            distance = metric(self.policy)
            if self.verbose:
                print(f'{nb_eff_search}: {distance}')
            if distance > max_distance:
                stepsize /= 2.
                self.set_weights(pol.parameters())
            else:
                return
