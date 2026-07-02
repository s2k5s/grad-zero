import torch

# original implementation involved making pairs of adjacent elements
# it was later realised that slicing tensors into interleaved adjacent 
# pairs can sometimes be unfriendly to GPU memory contiguity and certain CUDA kernels.

def apply_rope_adjacent(x_q, x_k, freq_cis):
    # q and k come from the same input x, so batch size and seq len always
    # match; only head count can differ (e.g. GQA: num_q_heads > num_kv_heads)
    B, num_hq, L, h_dim = x_q.shape
    _, num_hk, _, _ = x_k.shape
    xq = x_q.float().reshape(B, num_hq, L, h_dim//2, 2)
    xk = x_k.float().reshape(B, num_hk, L, h_dim//2, 2)
    xq_complex = torch.view_as_complex(xq)
    xk_complex = torch.view_as_complex(xk)

    #freq_cis has dim ->(L, h_dim//2)
    #(L, h_dim//2) -->(1, 1, L, h_dim//2)
    freq_cis = freq_cis.unsqueeze(0).unsqueeze(1)

    prod_q = xq_complex * freq_cis
    prod_k = xk_complex * freq_cis

    xq_real = torch.view_as_real(prod_q)
    xk_real = torch.view_as_real(prod_k)

    xq_out = xq_real.reshape(B, num_hq, L, h_dim).type_as(x_q)
    xk_out = xk_real.reshape(B, num_hk, L, h_dim).type_as(x_k)
    return xq_out, xk_out