import math
import torch
import torch_geometric as pyg
from datasets.data_utils import collate_fn
def loglik_nats(model, x):
    """Compute the log-likelihood in nats."""
    return - model.log_prob(x).mean()


def loglik_bpd(model, x):
    """Compute the log-likelihood in bits per dim."""
    return -model.log_prob(x).sum() / (math.log(2) * x.num_entries)
    # return - model.log_prob(x).sum() / (math.log(2) * x.shape.numel())


def elbo_nats(model, x):
    """
    Compute the ELBO in nats.
    Same as .loglik_nats(), but may improve readability.
    """
    return loglik_nats(model, x)


def elbo_bpd(model, x):
    """
    Compute the ELBO in bits per dim.
    Same as .loglik_bpd(), but may improve readability.
    """
    return loglik_bpd(model, x)


def iwbo(model, x, k):
    ll = -model.nll(x)
    # ll = torch.stack(torch.chunk(ll_stack, k, dim=0))
    # print(ll)
    return torch.logsumexp(ll, dim=0) - math.log(k)


# def iwbo_batched(model, x, k, kbs):
#     assert k % kbs == 0
#     num_passes = k // kbs
#     ll_batched = []
#     for i in range(num_passes):
#         x_stack = torch.cat([x for _ in range(kbs)], dim=0)
#         ll_stack = model.log_prob(x_stack)
#         ll_batched.append(torch.stack(torch.chunk(ll_stack, kbs, dim=0)))
#     ll = torch.cat(ll_batched, dim=0)
#     return torch.logsumexp(ll, dim=0) - math.log(k)


# def iwbo_nats(model, x, k, kbs=None):
#     """Compute the IWBO in nats."""
#     if kbs: return - iwbo_batched(model, x, k, kbs).mean()
#     else:   return - iwbo(model, x, k).mean()


# def iwbo_bpd(model, x, k, kbs=None):
#     """Compute the IWBO in bits per dim."""
#     if kbs: return - iwbo_batched(model, x, k, kbs).sum() / (x.numel() * math.log(2))
#     else:   return - iwbo(model, x, k).sum() / (x.numel() * math.log(2))


def dataset_elbo_nats(model, data_loader, device, double=False, verbose=True):
    with torch.no_grad():
        nats = 0.0
        count = 0
        for i, x in enumerate(data_loader):
            if double: x = x.double()
            x = x.to(device)
            nats += elbo_nats(model, x).cpu().item() * len(x)
            count += len(x)
            if verbose: print('{}/{}'.format(i+1, len(data_loader)), nats/count, end='\r')
    return nats / count


def dataset_elbo_bpd(model, data_loader, device, double=False, verbose=True):
    with torch.no_grad():
        bpd = 0.0
        count = 0
        for i, x in enumerate(data_loader):
            if double: x = x.double()
            x = x.to(device)
            bpd += elbo_bpd(model, x).cpu().item() * len(x)
            count += len(x)
            if verbose: print('{}/{}'.format(i+1, len(data_loader)), bpd/count, end='\r')
    return bpd / count


# def dataset_iwbo_nats(model, data_loader, k, device, double=False, kbs=None, verbose=True):
#     with torch.no_grad():
#         nats = 0.0
#         count = 0
#         for i, x in enumerate(data_loader):
#             if double: x = x.double()
#             x = x.to(device)
#             nats += iwbo_nats(model, x, k=k, kbs=kbs).cpu().item() * len(x)
#             count += len(x)
#             if verbose: print('{}/{}'.format(i+1, len(data_loader)), nats/count, end='\r')
#     return nats / count


# def dataset_iwbo_bpd(model, data_loader, k, device, double=False, kbs=None, verbose=True):
#     with torch.no_grad():
#         bpd = 0.0
#         count = 0
#         for i, x in enumerate(data_loader):
#             if double: x = x.double()
#             x = x.to(device)
#             bpd += iwbo_bpd(model, x, k=k, kbs=kbs).cpu().item() * len(x)
#             count += len(x)
#             if verbose: print('{}/{}'.format(i+1, len(data_loader)), bpd/count, end='\r')
#     return bpd / count
def _infer_num_entries(model, x):
    # 1) 이미 있으면 그걸 사용
    if hasattr(x, "num_entries"):
        ne = x.num_entries
        if torch.is_tensor(ne):
            ne = int(ne.item()) if ne.numel() == 1 else int(ne.sum().item())
        return int(ne)

    # 2) DP면 model.module, 아니면 model에서 _calc_num_entries 사용
    base_model = model.module if hasattr(model, "module") else model
    if hasattr(base_model, "_calc_num_entries"):
        ne = base_model._calc_num_entries(x)
        if torch.is_tensor(ne):
            ne = int(ne.item()) if ne.numel() == 1 else int(ne.sum().item())
        return int(ne)

    # 3) 최후의 fallback (둘 다 없을 때)
    ne = 0
    if hasattr(x, "full_edge_attr"):
        ne += int(x.full_edge_attr.shape[0])
    if hasattr(x, "node_attr"):
        ne += int(x.node_attr.shape[0])
    if ne <= 0:
        raise AttributeError("Cannot infer num_entries from x.")
    return ne


def loglik_bpd(model, x):
    """Compute the log-likelihood in bits per dim."""
    num_entries = _infer_num_entries(model, x)
    return -model.log_prob(x).sum() / (math.log(2) * float(num_entries))
