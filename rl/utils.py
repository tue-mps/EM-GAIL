import torch
from torch.autograd import grad
from torch.distributions import Independent
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal

def get_device():

    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

save_dir = 'saved-sessions'

def apply_update(parameterized_fun, update):

    n = 0

    for param in parameterized_fun.parameters():
        numel = param.numel()
        param_update = update[n:n + numel].view(param.size())
        param.data += param_update
        n += numel

def flatten(vecs):

    flattened = torch.cat([v.view(-1) for v in vecs])

    return flattened

def flat_grad(functional_output, inputs, retain_graph=False, create_graph=False):

    if create_graph == True:
        retain_graph = True

    grads = grad(functional_output, inputs, retain_graph=retain_graph, create_graph=create_graph)
    flat_grads = flatten(grads)

    return flat_grads

def get_flat_params(parameterized_fun):

    parameters = parameterized_fun.parameters()
    flat_params = flatten([param.view(-1) for param in parameters])

    return flat_params


def line_search(search_dir, max_step_len, constraints_satisfied, device, line_search_coef=0.9,
                max_iter=10):

    step_len = max_step_len / line_search_coef

    for i in range(max_iter):
        step_len *= line_search_coef

        if constraints_satisfied(step_len * search_dir, step_len):
            return step_len

    return torch.tensor(0.0).to(device)

def cg_solver(Avp_fun, b, max_iter=10):

    device = get_device()
    x = torch.zeros_like(b).to(device)
    r = b.clone()
    p = b.clone()

    for i in range(max_iter):
        Avp = Avp_fun(p, retain_graph=True)

        alpha = torch.matmul(r, r) / torch.matmul(p, Avp)
        x += alpha * p

        if i == max_iter - 1:
            return x

        r_new = r - alpha * Avp
        beta = torch.matmul(r_new, r_new) / torch.matmul(r, r)
        r = r_new
        p = r + beta * p



def detach_dist(dist):


    if type(dist) is Categorical:
        detached_dist = Categorical(logits=dist.logits.detach())
    elif type(dist) is Independent:
        detached_dist = Normal(loc=dist.mean.detach(), scale=dist.stddev.detach())
        detached_dist = Independent(detached_dist, 1)

    return detached_dist

def mean_kl_first_fixed(dist_1, dist_2):

    dist_1_detached = detach_dist(dist_1)
    mean_kl = torch.mean(kl_divergence(dist_1_detached, dist_2))

    return mean_kl



def get_Hvp_fun(functional_output, inputs, damping_coef=0.0):


    inputs = list(inputs)
    grad_f = flat_grad(functional_output, inputs, create_graph=True)

    def Hvp_fun(v, retain_graph=True):
        gvp = torch.matmul(grad_f, v)
        Hvp = flat_grad(gvp, inputs, retain_graph=retain_graph)
        Hvp += damping_coef * v

        return Hvp

    return Hvp_fun