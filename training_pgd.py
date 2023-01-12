import numpy as np
import simulation
import projected_gradient_descent as pgd
from datetime import datetime


# FSP gradient descent

def pgdFSP(crn, 
            ind_species, 
            domain, 
            fixed_params, 
            time_windows, 
            loss, 
            grad_loss, 
            cr, 
            gamma, 
            n_iter, 
            tolerance_rounds, 
            tolerance, 
            targets,
            crn_name, 
            weights=None,
            save=(True, ['control_values', 'experimental_losses', 'parameters', 'gradients_losses', 'real_losses', 'exp_results'])):
    optimizerFSP = pgd.ProjectedGradientDescent_FSP(crn=crn,
                                                    ind_species=ind_species,
                                                    domain=domain,
                                                    fixed_params=fixed_params,
                                                    time_windows=time_windows,
                                                    loss=loss,
                                                    grad_loss=grad_loss,
                                                    weights=weights,
                                                    cr=cr)
    final_time, control_params, loss_value = pgd.control_method(optimizer=optimizerFSP,
                                                                gamma=gamma,
                                                                n_iter=n_iter,
                                                                tolerance_rounds=tolerance_rounds,
                                                                tolerance=tolerance,
                                                                ind_species=ind_species,
                                                                targets=targets,
                                                                rate_performance=n_iter//10,
                                                                save=save)
    with open(f'data_pgdFS_{crn_name}_{datetime.now().strftime("%d.%m.%Y_%H.%M.%S")}.txt', 'w') as f:
        f.write(f'domain: {domain}\n')
        f.write(f'fixed parameters: {fixed_params}\n')
        f.write(f'time_windows: {time_windows}\n')
        f.write(f'loss: {loss}\n')
        f.write(fr'$c_r$: {cr}\n')
        f.write(f'gamma: {gamma}\n')
        f.write(f'n_iter: {n_iter}\n')
        f.write(f'tolerance_rounds: {tolerance_rounds}\n')
        f.write(f'targets: {targets}\n')
        f.write(f'PGD time: {final_time}\n')
        f.write(f'Final parameters: {control_params}\n')
        f.write(f'Final loss: {loss_value}')

def pgdMDN(crn, 
            model,
            domain, 
            fixed_params, 
            time_windows, 
            loss,
            gamma, 
            n_iter, 
            tolerance_rounds, 
            tolerance,
            ind_species, 
            targets,
            crn_name, 
            weights=None,
            directory="",
            save=(True, ['control_values', 'experimental_losses', 'parameters', 'gradients_losses', 'real_losses', 'exp_results'])):
    optimizerMDN = pgd.ProjectedGradientDescent_MDN(crn=crn,
                                                    model=model,
                                                    domain=domain,
                                                    fixed_params=fixed_params,
                                                    time_windows=time_windows,
                                                    loss=loss,
                                                    weights=weights)
    final_time, control_params, loss_value = pgd.control_method(optimizer=optimizerMDN,
                                                                gamma=gamma,
                                                                n_iter=n_iter,
                                                                tolerance_rounds=tolerance_rounds,
                                                                tolerance=tolerance,
                                                                ind_species=ind_species,
                                                                targets=targets,
                                                                rate_performance=n_iter//10,
                                                                save=save)
    with open(f'{directory}data_pgdMDN_{crn_name}_{datetime.now().strftime("%d.%m.%Y_%H.%M.%S")}.txt', 'w') as f:
        f.write(f'domain: {domain}\n')
        f.write(f'fixed parameters: {fixed_params}\n')
        f.write(f'time_windows: {time_windows}\n')
        f.write(f'loss: {loss(0)}, {loss(1)}, {loss(2)}\n')
        f.write(f'gamma: {gamma}\n')
        f.write(f'n_iter: {n_iter}\n')
        f.write(f'tolerance_rounds: {tolerance_rounds}\n')
        f.write(f'targets: {targets}\n')
        f.write(f'PGD time: {final_time}\n')
        f.write(f'Final parameters: {control_params}\n')
        f.write(f'Final loss: {loss_value}')



if __name__ == '__main__':

    from CRN4_control import propensities_bursting_gene as propensities
    import save_load_MDN

    def loss3(x):
        return (x-3)**2

    def loss2(x):
        return (x-2)**2

    def loss1(x):
        return (x-1)**2

    def loss4(x):
        return (x-0.5)**2

    def grad_loss3(probs, gradient):
        return 2*gradient*(probs-3)

    def grad_loss2(probs, gradient):
        return 2*gradient*(probs-2)

    def grad_loss1(probs, gradient):
        return 2*gradient*(probs-1)

    def grad_loss4(probs, gradient):
        return 2*gradient*(probs-0.5)

    crn = simulation.CRN(stoichiometry_mat=propensities.stoich_mat, 
                        propensities=propensities.propensities, 
                        propensities_drv=None, 
                        init_state=propensities.init_state, 
                        n_fixed_params=3, 
                        n_control_params=1)
    domain = np.stack([np.array([1e-5, 5.])]*4)
    fixed_params = np.array([1., 2., 1.])
    time_windows = np.array([5, 10, 15, 20])


    # pgdFSP(crn=crn,
    #         ind_species=propensities.ind_species,
    #         domain=domain,
    #         fixed_params=fixed_params,
    #         time_windows=time_windows,
    #         loss=loss1,
    #         grad_loss=grad_loss1,
    #         cr=50,
    #         gamma=0.1,
    #         n_iter=1_000,
    #         tolerance_rounds=15,
    #         tolerance=1e-7,
    #         targets=np.array([[5., 1.], [10., 1.], [15., 1.], [20., 1.]]),
    #         crn_name='CRN4')

    model = save_load_MDN.load_MDN_model('CRN4_control/saved_models/CRN4_model2.pt')

    pgdMDN(crn=crn,
            model=model,
            domain=domain,
            fixed_params=fixed_params,
            time_windows=time_windows,
            loss=loss4,
            gamma=0.01,
            n_iter=10_000,
            tolerance_rounds=100,
            tolerance=1e-7,
            ind_species=propensities.ind_species,
            targets=np.array([[5., 0.5], [10., 0.5], [15., 0.5], [20., 0.5]]),
            crn_name='CRN4')

