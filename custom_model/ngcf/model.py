from torch import nn
import torch

import numpy as np

from .args import Args
import scipy.sparse as sp
import torch.nn.functional as F

from .logger import get_logger, logging_conf

logger = get_logger(logger_conf=logging_conf)


class NGCF(nn.Module):
    def __init__(
        self,
        args: Args,
        device: torch.device,
        n_users: int,
        n_items: int,
        norm_adj: sp.csr_array,
    ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.device = device

        # hyperparameters
        self.node_dropout = args.node_dropout
        self.mess_dropout = args.mess_dropout
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
                        torch.empty(
                            self.n_users, self.embedding_dim, device=self.device
                        )
                    )
                ),
                "item_embedding": nn.Parameter(
                    initializer(
                        torch.empty(
                            self.n_items, self.embedding_dim, device=self.device
                        )
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
        random_tensor = 1 - self.node_dropout
        random_tensor += torch.rand(x._nnz(), device=self.device)
        print(random_tensor)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        out = torch.sparse.FloatTensor(i, v, x.shape).to(self.device)
        return out * (1.0 / (1 - self.node_dropout))

    def forward(self, u, i, j):
        A_hat = self.sparse_dropout(self.sparse_norm_adj)

        ego_embeddings = torch.cat(
            [self.embedding_dict["user_emb"], self.embedding_dict["item_emb"]],
            0,
        )

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            sum_embeddings = (
                torch.matmul(side_embeddings, self.weight_dict["W_gc_%d" % k])
                + self.weight_dict["b_gc_%d" % k]
            )

            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)

            bi_embeddings = (
                torch.matmul(bi_embeddings, self.weight_dict["W_bi_%d" % k])
                + self.weight_dict["b_bi_%d" % k]
            )

            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(
                sum_embeddings + bi_embeddings
            )

            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[: self.n_users, :]
        i_g_embeddings = all_embeddings[self.n_users :, :]

        u_g_embeddings = u_g_embeddings[u, :]
        pos_i_g_embeddings = i_g_embeddings[i, :]
        neg_i_g_embeddings = i_g_embeddings[j, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
