"""Fixed movie features and profiles computed from training interactions only."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from task1.model_registry import PROJECT_ROOT


def movie_features(dataset, config):
    path = Path(config["metadata_path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Movie metadata missing: {path}. See README feature combination setup.")
    movies = pd.read_csv(path, sep="\t", names=["item_id", "title", "genre", "description"],
                         dtype=str, keep_default_na=False).set_index("item_id")
    tokens = dataset.id2token(dataset.iid_field, np.arange(dataset.item_num))
    missing = set(tokens[1:]) - set(movies.index)
    if missing:
        raise ValueError(f"Metadata missing for {len(missing)} movie IDs")
    movies = movies.reindex(tokens).fillna("")
    genres = sorted({g.strip() for value in movies.genre for g in value.split(",") if g.strip()})
    genre_features = np.array([[float(g in {x.strip() for x in value.split(",")})
                               for g in genres] for value in movies.genre], dtype=np.float32)
    # This cache contains item metadata only; user profiles are rebuilt for every split.
    key = hashlib.sha256(path.read_bytes() + str((list(tokens), config["text_dimension"],
                                                config["seed"])).encode()).hexdigest()[:16]
    cache = PROJECT_ROOT / "artifacts" / "features" / f"text-{key}.npy"
    if cache.exists():
        text_features = np.load(cache)
    else:
        text = movies.title + " " + movies.description
        matrix = TfidfVectorizer(max_features=5000, stop_words="english").fit_transform(text)
        dimension = min(config["text_dimension"], matrix.shape[0] - 1, matrix.shape[1] - 1)
        text_features = normalize(TruncatedSVD(n_components=dimension,
                                  random_state=config["seed"]).fit_transform(matrix)).astype(np.float32)
        text_features[0] = 0
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, text_features)
    return genre_features, text_features


def build_features(dataset, config):
    genres, text = movie_features(dataset, config)
    interactions = dataset.inter_matrix(form="csr").astype(np.float32)
    counts = np.asarray(interactions.sum(axis=1)).reshape(-1, 1)
    profiles = lambda features: np.asarray(interactions @ features) / np.maximum(counts, 1)
    popularity = np.log1p(np.asarray(interactions.sum(axis=0)).reshape(-1, 1))
    popularity /= max(float(popularity.max()), 1)
    arrays = {
        "item_genres": genres, "user_genres": profiles(genres),
        "item_text": text, "user_text": profiles(text), "popularity": popularity,
    }
    return {name: torch.from_numpy(np.asarray(value, dtype=np.float32))
            for name, value in arrays.items()}
