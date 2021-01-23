"""
Arxiv: 1805.11593
"""
import torch


def transformed_h(qval: torch.tensor, eps: float = 1e-2):
    return qval.sign() * ((qval.abs() + 1).sqrt() - 1) + eps * qval


def transformed_h_reverse(qval: torch.tensor, eps: float = 1e-2):
    return qval.sign() * (
        (((1 + 4 * eps * (qval.abs() + 1 + eps)).sqrt() - 1) / (2 * eps)).pow(2) - 1
    )
