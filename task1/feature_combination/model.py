"""A single pairwise recommender combining interaction and metadata features."""
import torch
from torch import nn
from torch.nn import functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType
from task1.feature_combination.features import build_features


class FeatureCombination(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.use_genres = config["use_genres"]
        self.use_text = config["use_text"]
        self.use_popularity = config["use_popularity"]
        self.reg_weight = config["reg_weight"]
        features = build_features(dataset, config)
        for name, values in features.items():
            self.register_buffer(name, values)
        dimension = config["embedding_size"]
        self.user_embedding = nn.Embedding(self.n_users, dimension, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items, dimension, padding_idx=0)
        # The elementwise product makes user/item interaction explicit.
        size = 3 * dimension
        size += 2 * self.item_genres.shape[1] if self.use_genres else 0
        size += 2 * self.item_text.shape[1] if self.use_text else 0
        size += 1 if self.use_popularity else 0
        self.network = nn.Sequential(nn.Linear(size, config["hidden_size"]), nn.ReLU(),
                                     nn.Linear(config["hidden_size"], 1))
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)

    def forward(self, users, items):
        user = self.user_embedding(users)
        item = self.item_embedding(items)
        parts = [user, item, user * item]
        if self.use_genres:
            parts.extend([self.user_genres[users], self.item_genres[items]])
        if self.use_text:
            parts.extend([self.user_text[users], self.item_text[items]])
        if self.use_popularity:
            parts.append(self.popularity[items])
        return self.network(torch.cat(parts, dim=-1)).squeeze(-1)

    def calculate_loss(self, interaction):
        users = interaction[self.USER_ID]
        positive = interaction[self.ITEM_ID]
        negative = interaction[self.NEG_ITEM_ID]
        loss = F.softplus(self(users, negative) - self(users, positive)).mean()
        regularization = sum(parameter.square().sum() for parameter in self.parameters())
        return loss + self.reg_weight * regularization

    def predict(self, interaction):
        return self(interaction[self.USER_ID], interaction[self.ITEM_ID])

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        items = torch.arange(self.n_items, device=users.device)
        # Small chunks keep full-catalog scoring memory modest on CPU.
        scores = []
        for chunk in items.split(256):
            scores.append(self(users.repeat_interleave(len(chunk)), chunk.repeat(len(users)))
                          .view(len(users), -1))
        return torch.cat(scores, dim=1).reshape(-1)
