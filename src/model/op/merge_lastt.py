
import torch


class MergeLastToken(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inter_out, intra_out, inter_lse, intra_lse): # inter_lse and intra_lse should be in float32
        max_lse = torch.max(inter_lse, intra_lse)
        inter_lse_exp = torch.exp(inter_lse - max_lse)
        intra_lse_exp = torch.exp(intra_lse - max_lse)
        intra_adjust = (intra_lse_exp / (intra_lse_exp + inter_lse_exp)).to(intra_out.dtype).unsqueeze(-1)
        inter_adjust = (inter_lse_exp / (intra_lse_exp + inter_lse_exp)).to(inter_out.dtype).unsqueeze(-1)
        out = inter_out * inter_adjust + intra_adjust * intra_out
        ctx.save_for_backward(inter_out, intra_out, inter_adjust, intra_adjust)
        return out
 
    @staticmethod
    def backward(ctx, grad_output):
        inter_out, intra_out, inter_adjust, intra_adjust = ctx.saved_tensors
        grad_inter_lse = (grad_output * (inter_out - intra_out) * intra_adjust * inter_adjust).sum(-1)
        return grad_output * inter_adjust, grad_output * intra_adjust, grad_inter_lse, -grad_inter_lse


def merge_last_token_naive(inter_out, intra_out, inter_lse, intra_lse):
    max_lse = torch.max(inter_lse, intra_lse)
    inter_lse_exp = torch.exp(inter_lse - max_lse)
    intra_lse_exp = torch.exp(intra_lse - max_lse)
    intra_adjust = (intra_lse_exp / (intra_lse_exp + inter_lse_exp)).to(intra_out.dtype).unsqueeze(-1)
    inter_adjust = (inter_lse_exp / (intra_lse_exp + inter_lse_exp)).to(inter_out.dtype).unsqueeze(-1)
    return inter_out * inter_adjust + intra_adjust * intra_out


def merge_last_token_naive_unsafe(inter_out, intra_out, inter_lse, intra_lse):
    score_adjust = (1.0 / (1.0 + torch.exp(inter_lse - intra_lse))).to(intra_out.dtype).unsqueeze(-1)
    return inter_out * (1.0 - score_adjust) + score_adjust * intra_out


merge_last_token = MergeLastToken.apply
