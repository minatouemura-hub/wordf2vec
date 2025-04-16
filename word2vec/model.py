import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa

from word2vec.preprocessor import BookDataset


# 文脈-user_id , 単語 - 本のタイトル(著者)
# item2vecに似ているらしい
# 埋め込み表現
class UserBook2Vec(nn.Module):
    def __init__(
        self,
        dataset: "BookDataset" = None,
        embedding_dim: int = 50,
        num_negatives: int = 5,
        batch_size: int = 4,
        epochs: int = 10,
        learning_rate: int = 0.005,
    ):
        # super().__init__()
        # まず nn.Module の初期化を行う
        nn.Module.__init__(self)
        # BookDataset の初期化も行う（UserBook2Vec が BookDataset を継承しているので）
        self.__dict__.update(dataset.__dict__)

        self.num_users = len(self.user2id)
        self.vocab_size = len(self.book2id)

        # ユーザーの埋め込み（文脈側）
        self.user_embed = nn.Embedding(self.num_users, embedding_dim)
        # 書籍の埋め込み（ターゲット側）
        self.book_embed = nn.Embedding(self.vocab_size, embedding_dim)

        # 初期設定
        self.embedding_dim = embedding_dim
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

    def forward(self, user_ids, pos_book_ids, neg_book_ids):
        # user_ids: (batch_size,)
        # pos_book_ids: (batch_size,)
        # neg_book_ids: (batch_size, num_negatives)
        user_vec = self.user_embed(user_ids)  # (batch_size, embedding_dim)
        pos_book_vec = self.book_embed(pos_book_ids)  # (batch_size, embedding_dim)
        neg_book_vec = self.book_embed(neg_book_ids)  # (batch_size, num_negatives, embedding_dim)

        # 正例スコア（内積）
        pos_score = torch.sum(user_vec * pos_book_vec, dim=1)  # (batch_size,)
        pos_score = torch.log(torch.sigmoid(pos_score) + 1e-10)

        # 負例スコア
        neg_score = torch.bmm(neg_book_vec, user_vec.unsqueeze(2)).squeeze(
            -1
        )  # (batch_size, num_negatives)
        neg_score = torch.log(torch.sigmoid(-neg_score) + 1e-10)
        neg_score = torch.sum(neg_score, dim=1)  # (batch_size,)

        loss = -(pos_score + neg_score)
        return loss.mean()

    def get_negative_samples(self, vocab_size, book_freq, batch_size):
        negatives = np.random.choice(vocab_size, size=(batch_size, self.num_negatives), p=book_freq)
        return negatives
