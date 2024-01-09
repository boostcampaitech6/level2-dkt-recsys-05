import os
from .args import Args, Mode
from torch.utils.data import Dataset
import polars as pl
import torch
import scipy.sparse as sp
import numpy as np

from .logger import get_logger, logging_conf

logger = get_logger(logger_conf=logging_conf)


class NGCFDataset(Dataset):
    def __init__(self, args: Args):
        self.mode = args.mode
        self.data_dir = args.data_dir
        self.train_file = os.path.join(args.data_dir, "train_data.csv")
        self.test_file = os.path.join(args.data_dir, "test_data.csv")

        self.n_users, self.n_items = 0, 0

        train_df = pl.read_csv(self.train_file)
        test_df = pl.read_csv(self.test_file)

        # vstack을 사용함으로써 메모리를 
        all_df = train_df.vstack(test_df)

        self.n_users = all_df.select(pl.col("userID")).n_unique()
        self.n_items = all_df.select(pl.col("assessmentItemID")).n_unique()

        item_id_to_idx = {
            item_id: int(idx) + int(self.n_users)
            for idx, item_id in enumerate(
                all_df.select(pl.col("assessmentItemID")).unique().to_series().to_list()
            )
        }

        # 긍정 데이터만 사용한다
        # 이유: graph를 만들 때 긍정적인 상호작용(정답)으로 연결되어있다고 설계했기 때문
        # 부정적인 상호작용(틀림)은 맞출 확률과 멀어지는 것이기 때문에 연결되어있지 않다고 설계했기 때문
        positive_df = all_df.filter(pl.col("answerCode") == 1)

        self.features = torch.tensor(
            data=positive_df.with_columns(
                [
                    (pl.col("userID")).alias("user"),
                    (pl.col("assessmentItemID").replace(item_id_to_idx)).alias("item"),
                ]
            )
            .select(
                pl.col("user").cast(pl.Int64),
                pl.col("item").cast(pl.Int64),
            )
            .to_numpy(),
            
            dtype=torch.float32,
        )

        self.labels = torch.tensor(
            data=positive_df.select(pl.col("answerCode").alias("label")).to_numpy().reshape(-1),
            dtype=torch.float32,
        )

        assert len(self.features) == len(self.labels)

    def negative_pool(self):
        """
        Create Negative Pool
        """
        logger.info("Creating Negative Pool")
        negative_pool = set()
        for user, item in self.features:
            negative_pool.add((user, item))

        return negative_pool

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        :return: (user, positive_item, negative_item, positive_label, negative_label)
        """
        features = self.features[idx]

        return self.features[idx], self.labels[idx]

    def get_n_users(self):
        return self.n_users

    def get_n_items(self):
        return self.n_items

    def get_cached_adj_matrix(self) -> sp.csr_array:
        try:
            adj_matrix = sp.load_npz(os.path.join(self.data_dir, "adj_matrix.npz"))
            logger.info("Loaded Cached Adjacency Matrix")

        except FileNotFoundError:
            logger.info("Cached Adjacency Matrix Not Found. Creating New One")
            adj_matrix = self.create_adj_matrix()
            sp.save_npz(os.path.join(self.data_dir, "adj_matrix.npz"), adj_matrix)

        return adj_matrix

    def create_adj_matrix(self) -> sp.csr_array:
        """
        Create Adjacency Matrix
        """

        adj_matrix = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        adj_matrix: sp.lil_matrix = adj_matrix.tolil()

        R: sp.lil_matrix = sp.dok_matrix(
            (self.n_users, self.n_items), dtype=np.float32
        ).tolil()

        adj_matrix[: self.n_users, self.n_users :] = R
        adj_matrix[self.n_users :, : self.n_users] = R.T
        adj_matrix = adj_matrix.todok()

        # normalized adjacency matrix
        rowsum = np.array(adj_matrix.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj: sp.dia_matrix = d_mat_inv.dot(adj_matrix)

        return norm_adj.tocoo().tocsr()
