import torch
import neuralnetwork


def sensitivities(input, model, length_output=142):
    def f(input, length_output=length_output):
        layer_ww, layer_rr, layer_pp = model.forward(input)
        mat_k = torch.arange(length_output).repeat(1,model.n_comps,1).permute([2,0,1])
        return neuralnetwork.mix_nbpdf(layer_rr, layer_pp, layer_ww, mat_k)
    return torch.squeeze(torch.autograd.functional.jacobian(f, input))

def expected_val(input, model, length_output=142):
    sensitivity = sensitivities(input, model, length_output)
    expec = sensitivity.permute(1,0) * torch.arange(length_output)
    return expec.sum(dim=1)


