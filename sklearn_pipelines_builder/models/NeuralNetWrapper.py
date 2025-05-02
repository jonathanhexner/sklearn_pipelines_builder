import numpy as np
import math
import gc
import pandas as pd
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn_pipelines_builder.infrastructure.BaseConfigurableTransformer import BaseConfigurableTransformer
from sklearn_pipelines_builder.utils.basic_utils import get_features
from sklearn_pipelines_builder.utils.basic_utils import log_memory


from torch.utils.data import Dataset

class LazyDataset(Dataset):
    def __init__(self, X, y, numeric_cols, categorical_cols, scaler, label_encoders):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.scaler = scaler
        self.label_encoders = label_encoders

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X.iloc[idx]
        x_num = self.scaler.transform(pd.DataFrame([row[self.numeric_cols]]))[0]
        x_cat = [self.label_encoders[col].transform([row[col]])[0] for col in self.categorical_cols]
        y_val = self.y.iloc[idx]
        return (
            torch.tensor(x_num, dtype=torch.float32),
            *[torch.tensor(c, dtype=torch.long) for c in x_cat],
            torch.tensor(y_val, dtype=torch.float32)
        )


class SimpleFeedforwardNNWithEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config["input_dim"]
        embedding_dims = config["embedding_dims"]
        hidden_layers = config.get("hidden_layers", [64, 32])
        dropout = config.get("dropout", 0.2)
        batchnorm = config.get("batchnorm", True)

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, emb_dim) for num_categories, emb_dim in embedding_dims
        ])
        self.embedding_output_dim = sum([emb_dim for _, emb_dim in embedding_dims])
        total_input_dim = input_dim + self.embedding_output_dim

        layers = []
        dim = total_input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(dim, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, numeric_inputs, categorical_inputs):
        embeddings = []
        for emb, cat in zip(self.embeddings, categorical_inputs):
            cat = cat.long()
            cat = torch.clamp(cat, max=emb.num_embeddings - 1)
            embeddings.append(emb(cat))
        x = torch.cat(embeddings + [numeric_inputs], dim=1)
        return self.network(x).squeeze(-1)


class NeuralNetWrapper(BaseConfigurableTransformer):
    def __init__(self, config):
        self.config = config
        self.model_config = config.get("model_config", {})
        self.min_embedding_dim = self.model_config.get("min_embedding_dim", 20)
        self.shuffle = self.model_config.get('shuffle', False)
        self.batch_size = self.model_config.get("batch_size", 128)
        self.epochs = self.model_config.get("epochs", 50)
        self.patience = self.model_config.get("patience", 5)
        self.learning_rate = self.model_config.get("learning_rate", 1e-3)
        self.num_threads = self.model_config.get("num_threads", os.cpu_count())
        torch.set_num_threads(self.num_threads)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.feature_names = []
        self.numeric_cols = []
        self.categorical_cols = []
        self.label_encoders = {}
        self.category_sizes = []
        self.scaler = StandardScaler()
        self.classes_ = []

    # def _prepare_data(self, X, y):
    #     dataset = LazyDataset(
    #         X, y,
    #         self.numeric_cols,
    #         self.categorical_cols,
    #         self.scaler,
    #         self.label_encoders
    #     )
    #     return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=0)

    def _prepare_data(self, X, y):
        X_num = self.scaler.transform(X[self.numeric_cols]).astype(np.float32)
        X_cat = [le.transform(X[col]) for col, le in self.label_encoders.items()]
        y_vals = y.astype(np.float32).values

        X_num_tensor = torch.from_numpy(X_num)
        X_cat_tensors = [torch.from_numpy(col.astype(np.int64)) for col in X_cat]
        y_tensor = torch.from_numpy(y_vals)

        dataset = TensorDataset(X_num_tensor, *X_cat_tensors, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    def fit(self, X, y):
        self.feature_names = get_features(X)
        self.categorical_cols = X[self.feature_names].select_dtypes(include=['object', 'category']).columns.tolist()
        self.numeric_cols = [col for col in self.feature_names if col not in self.categorical_cols]

        # Fit encoders and scaler
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoders[col] = le
            embedding_dim = min(32, int(math.ceil(len(le.classes_) ** 0.25)))
            # embedding_dim = self.model_config.get("embedding_dim", max(self.min_embedding_dim, min(50, (len(le.classes_) + 1) // 2)))
            self.category_sizes.append((len(le.classes_), embedding_dim))
        log_memory("Before scaler fit")
        self.scaler.fit(X[self.numeric_cols])
        log_memory("After scaler fit")
        # Prepare data
        X[self.numeric_cols] = X[self.numeric_cols].astype(np.float32)
        gc.collect()
        log_memory("After casting to float32")


        loader = self._prepare_data(X, y)
        input_dim = len(self.numeric_cols)
        log_memory("Before model creation")
        model = SimpleFeedforwardNNWithEmbeddings({
            "input_dim": input_dim,
            "embedding_dims": self.category_sizes,
            "hidden_layers": self.model_config.get("hidden_layers", [32, 16]),
            "dropout": self.model_config.get("dropout", 0.2),
            "batchnorm": self.model_config.get("batchnorm", True)
        }).to(self.device)
        log_memory("After model creation")
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        best_loss = float("inf")
        patience_counter = 0

        from sklearn_pipelines_builder.utils.logger import logger

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            for n_batch, batch in enumerate(loader):
                if n_batch % 100 == 0:
                    logger.info(f"Epoch {epoch + 1}/{self.epochs} - Batch {n_batch}/{len(loader)}")
                    log_memory(f"Memory usage during training epoch {epoch} batch {n_batch}")
                xb, *cat_inputs, yb = batch
                cat_inputs = [c.to(self.device) for c in cat_inputs]
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                preds = model(xb, cat_inputs)
                loss = F.mse_loss(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)

            avg_loss = total_loss / len(loader.dataset)
            logger.info(f"Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self.model = model
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        X_num = torch.tensor(self.scaler.transform(X[self.numeric_cols]), dtype=torch.float32).to(self.device)
        X_cat = []
        for col in self.categorical_cols:
            le = self.label_encoders[col]
            known = set(le.classes_)
            mapped = X[col].map(lambda x: x if x in known else le.classes_[0])
            X_cat.append(torch.tensor(le.transform(mapped), dtype=torch.long).to(self.device))
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_num, X_cat).cpu().numpy()
        return preds

    def transform(self, X):
        return self.predict(X)

    def __call__(self, X):
        return self.predict(X)
