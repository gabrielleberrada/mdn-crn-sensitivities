import numpy as np
from casadi import *
import matplotlib.pyplot as plt


class ProjectedGradientDescent():
    def __init__(self, grad_f, domain, dim):
        """
        Inputs: 'grad_f'
                'domain': domain for the projection. Dim (2, N) such that 
                domain[0,:] defines the lower boundaries for each dimension and 
                domain[1,:] defines the upper boundaries for each dimension.
                'dim': dimension of the space
        """
        self.grad_f = grad_f
        self.domain = domain
        self.dim = dim


    def projected_gradient_descent(self, init, gamma, nb_iter=1_000, tolerance=1e-20, tolerance_rounds=20, norm=casadi.norm_2):
        xt_param1 = [init[0]]
        xt_param2 = [init[1]]
        xt = [init]
        costs = []
        for i in range(1, nb_iter + 1):
            opti = casadi.Opti()
            y = xt[-1] - gamma(xt[-1])*self.grad_f(xt[-1])
            costs.append(gamma(xt[-1]))
            x = opti.variable(self.dim)
            opti.minimize(norm(x - y))
            for n in range(self.dim):
                opti.subject_to(x[n] > self.domain[0,n])
                opti.subject_to(x[n] < self.domain[1,n])
            opti.solver('ipopt');
            sol = opti.solve();
            xt1 = sol.value(x)
            xt_param1.append(xt1[0])
            xt_param2.append(xt1[0])
            xt.append(xt1)
            last = i
        print('Number of iterations:', last)
        return xt_param1, xt_param2, costs



# testing

t_1 = 3
t_2 = 7
target = 20
theta = 5

def cost(params):
    xsi_1, xsi_2 = params[0], params[1]
    return (theta + xsi_1)**2 * t_1**3 / 3 + (1 - 2 * target) * (theta + xsi_1) / 2 * t_1**2 + target**2 * t_1 + ((theta + xsi_1) * t_1 + (theta + xsi_1)**2 * t_1**2) * (t_2 - t_1) + (2 * (xsi_1 + theta) * t_1 + 1) * (theta + xsi_2) * (t_2 - t_1)**2 / 2 + (theta + xsi_2)**2 * (t_2 - t_1)**3 / 3 - 2 * target * ((xsi_1 + theta) * t_1 * (t_2 - t_1) + (theta + xsi_2) * (t_2 - t_1)**2 / 2) + target**2 * (t_2 - t_1)

def grad_f(params):
    xsi_1, xsi_2 = params[0], params[1]
    grad1 = t_1**3 / 3 * 2 * (theta + xsi_1) + (1 - 2 * target) * t_1**2 / 2 + (t_2 - t_1) * (t_1 + t_1**2 * 2 * (theta + xsi_1)) + (theta + xsi_2) * (t_2 - t_1)**2 * t_1 - 2 * target * t_1 * (t_2 - t_1)
    grad2 = (2 * (xsi_1 + theta) * t_1 + 1) * (t_2 - t_1)**2 / 2 + (t_2 - t_1)**3 / 3 * 2 * (xsi_2 + theta) - 2 * target * (t_2 - t_1)**2 / 2
    return np.array([grad1, grad2])

domain = np.array([[0, 0], [5, 5]])

pgd = ProjectedGradientDescent(grad_f, domain, 2)
xt_param1, xt_param2, yt = pgd.projected_gradient_descent(init=np.array([2, 6]), nb_iter=20, gamma=cost)


print(xt_param1[-1], xt_param2[-1], yt[-1])
plt.plot(xt_param1)
plt.ylabel('Parameter 1')
plt.xlabel('Iterations')
plt.title('Gradient descent')
plt.show()
plt.plot(xt_param2, label='Parameter 2')
plt.xlabel('Iterations')
plt.ylabel('Parameter 2')
plt.title('Gradient descent')
plt.show()
plt.plot(yt)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient descent')
plt.show()




