import torch
import neuralnetwork


def sensitivities(input, model, length_output=142, with_probs=False):
    def f(input, length_output=length_output):
        mat_k = torch.arange(length_output).repeat(1,model.n_comps,1).permute([2,0,1])
        return neuralnetwork.mix_nbpdf(model, input, mat_k)
    if with_probs:
        return torch.squeeze(torch.autograd.functional.jacobian(f, input)), f(input).detach()
    return torch.squeeze(torch.autograd.functional.jacobian(f, input))

def expected_val(input, model, length_output=142):
    sensitivity = sensitivities(input, model, length_output)
    expec = sensitivity.permute(1,0) * torch.arange(length_output)
    return expec.sum(dim=1)


