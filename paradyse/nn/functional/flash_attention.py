import torch
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward


class FlashAttnKVPackedFunc():
    """
    Arguments:
    q: (batch_size, seqlen, nheads, headdim)
    kv: (2, batch_size, seqlen, nheads_k, headdim)
    dropout_p: float. Dropout probability.
    softmax_scale: float. The scaling of QK^T before applying softmax.
        Default to 1 / sqrt(headdim).
    causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
    window_size: (left, right). If not (-1, -1), implements sliding window local attention.
    alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
        (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
        is added to the attention score of query i and key j.
    deterministic: bool. Whether to use the deterministic implementation of the backward pass,
        which is slightly slower and uses more memory. The forward pass is always deterministic.
    return_attn_probs: bool. Whether to return the attention probabilities. This option is for
       testing only. The returned probabilities are not guaranteed to be correct
       (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept)

    """

    @staticmethod
    def forward(
            q,
            kv,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
            alibi_slopes=None,
            return_softmax=False,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q,
            kv[0],
            kv[1],
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0,
            # window_size=window_size,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
        )

        return out, softmax_lse, S_dmask, rng_state

    @staticmethod
    def backward(dout, q, kv, out, softmax_lse, rng_state):
        # q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq = torch.empty_like(q)
        scale = q.shape[-1] ** (-0.5)
        kv_shape = kv.shape
        dkv = torch.empty(kv_shape, dtype=kv.dtype, device=kv.device)
        k = kv[0]
        v = kv[1]
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dkv[0],
            dkv[1],
            dropout_p=0,
            softmax_scale=scale,
            causal=False,
            window_size_right=-1,
            window_size_left=-1,
            softcap=0,
            # window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dkv = dkv[..., : dout.shape[-1]]
        return dq, dkv  # , None, None, None, None, None, None, None
