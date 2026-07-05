import torch

# original implementation involved making pairs of adjacent elements
# it was later realised that slicing tensors into interleaved adjacent 
# pairs can sometimes be unfriendly to GPU memory contiguity and certain CUDA kernels.

def apply_rope_adjacent(x_q, x_k, freq_cis):
    """q and k can have different head counts (e.g. GQA), so their shapes
    are derived independently rather than assuming they match
    during inference as we have single query tensor and stacked key tensors
    the dimension of sequence length can differ, similarly num_heads
    also change in GQA"""
    Bq, num_hq, Lq, h_dim = x_q.shape
    Bk, num_hk, Lk, h_dim = x_k.shape
    xq = x_q.float().reshape(Bq, num_hq, Lq, h_dim//2, 2)
    xk = x_k.float().reshape(Bk, num_hk, Lk, h_dim//2, 2)
    xq_complex = torch.view_as_complex(xq)
    xk_complex = torch.view_as_complex(xk)

    #freq_cis has dim ->(L, h_dim//2)
    #(L, h_dim//2) -->(1, 1, L, h_dim//2)
    freq_cis = freq_cis.unsqueeze(0).unsqueeze(1)

    prod_q = xq_complex * freq_cis
    prod_k = xk_complex * freq_cis

    xq_real = torch.view_as_real(prod_q)
    xk_real = torch.view_as_real(prod_k)

    xq_out = xq_real.reshape(Bq, num_hq, Lq, h_dim).type_as(x_q)
    xk_out = xk_real.reshape(Bk, num_hk, Lk, h_dim).type_as(x_k)
    return xq_out, xk_out


def compute_freq_cis(head_dim, max_seq_length, theta=10000.0):
    """
    θ_i = 1 / (theta ^ (2i / head_dim))
    m : along the sequence
    stack the pairs of cos(mθ_i) and sin(mθ_i)
    """
    indices = torch.arange(0, head_dim, 2).float()
    theta_i = 1.0 / (theta ** (indices / head_dim))
    m = torch.arange(0, max_seq_length).float()
    angles = torch.outer(m, theta_i)# the dimension we want is rows is the input dim
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    freqs_real = torch.stack([cos, sin], dim=-1) #Go to the smallest atomic unit (the individual numbers) and interleave them directly
    freq_cis_complex = torch.view_as_complex(freqs_real)
    return freq_cis_complex
