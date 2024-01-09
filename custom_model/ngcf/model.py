from torch import nn
import torch

import numpy as np

from .args import Args
import scipy.sparse as sp
from torch.utils.data import DataLoader

from .logger import get_logger, logging_conf

logger = get_logger(logger_conf=logging_conf)


class NGCF(nn.Module):
    def __init__(self, args: Args, device: torch.device, n_users: int, n_items: int, norm_adj: sp.csr_array):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.device = device

        # hyperparameters
        self.dropout = args.dropout
        self.embedding_dim = args.embedding_dim
        self.layers = args.layers
        self.lr = args.lr
        self.norm_adj = norm_adj

        self.n_layers = len(args.layers)

        # initialize weights
        self.embedding_dict, self.weight_dict = self._init_weight()

        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(norm_adj)

    def _init_weight(self):
        """
        웨이트를 초기화한다.

        :return:
        """
        logger.info("Initializing Weights")
        # xavier_uniform_
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict(
            {
                "user_embedding": nn.Parameter(
                    initializer(
                        torch.empty(self.n_users, self.embedding_dim, device=self.device)
                    )
                ),
                "item_embedding": nn.Parameter(
                    initializer(
                        torch.empty(self.n_items, self.embedding_dim, device=self.device)
                    )
                ),
            }
        )

        weight_dict = nn.ParameterDict()
        layers = [self.embedding_dim] + self.layers

        for i in range(self.n_layers):
            # gc: graph convolution, bi: bias
            weight_dict[f"W_gc_{i}"] = nn.Parameter(
                initializer(
                    torch.empty(
                        layers[i],
                        layers[i + 1],
                    )
                )
            )

            weight_dict[f"W_bi_{i}"] = nn.Parameter(
                initializer(
                    torch.empty(
                        layers[i],
                        layers[i + 1],
                    )
                )
            )

            weight_dict[f"b_gc_{i}"] = nn.Parameter(
                initializer(
                    torch.empty(
                        1,
                        layers[i + 1],
                    )
                )
            )

            weight_dict[f"b_bi_{i}"] = nn.Parameter(
                initializer(
                    torch.empty(
                        1,
                        layers[i + 1],
                    )
                )
            )

        return embedding_dict, weight_dict
    
    def _convert_sp_mat_to_sp_tensor(self, X: sp.coo_matrix) -> torch.FloatTensor:
        """
        Convert scipy sparse matrix to PyTorch sparse matrix

        Arguments:
        ----------
        X = Adjacency matrix, scipy sparse matrix
        """
        coo = X.tocoo().astype(np.float32)
        i = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)
        res = torch.sparse.FloatTensor(i, v, coo.shape).to(self.device)
        return res
    
    def sparse_dropout(
        self,
        x: torch.FloatTensor,
    ):
        random_tensor = 1 - self.dropout
        random_tensor += torch.rand(x._nnz(), device=self.device)
        print(random_tensor)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        out = torch.sparse.FloatTensor(i, v, x.shape).to(self.device)
        return out * (1. / (1 - self.dropout))

    def forward(self, u, i, j):
        A_hat = self.sparse_dropout(self.sparse_norm_adj)