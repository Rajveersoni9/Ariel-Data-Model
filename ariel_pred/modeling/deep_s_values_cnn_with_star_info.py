import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from tqdm.auto import tqdm


class ResNetBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        out += identity
        out = self.relu(out)
        return out


class DeepSValuesCNNWithStarInfoTorch(nn.Module):
    def __init__(
        self,
        num_channels=6,
        wavelengths=283,
        num_star_features=3,
        num_resnet_blocks=10,
        resnet_hidden=1024,
        cnn_hidden_channels=256,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.wavelengths = wavelengths
        self.num_star_features = num_star_features

        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, cnn_hidden_channels, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_channels, cnn_hidden_channels, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_channels, cnn_hidden_channels, kernel_size=7, padding="same"),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_channels, 1, kernel_size=9, padding="same"),
            nn.Flatten(),
        )
        self.resnet = nn.Sequential(
            nn.Linear(wavelengths + num_star_features, resnet_hidden),
            nn.ReLU(),
            *[ResNetBlock(resnet_hidden) for _ in range(num_resnet_blocks)],
            nn.Linear(resnet_hidden, 1024),
            nn.Linear(1024, wavelengths),
        )

    def forward(self, s_values, star_info):
        # s_values: (batch_size, num_channels, wavelengths)
        x = self.cnn(s_values)  # (batch_size, wavelengths)
        # star_info: (batch_size, num_star_features)
        # Concatenate along feature axis
        x = torch.cat([x, star_info], dim=1)  # (batch_size, wavelengths + num_star_features)

        x = self.resnet(x)  # (batch_size, wavelengths)
        return x


class _SValuesDataset(torch.utils.data.Dataset):
    def __init__(self, s_values, star_info, targets=None):
        self.s_values = torch.tensor(s_values, dtype=torch.float32)
        self.star_info = torch.tensor(star_info, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None

    def __len__(self):
        return len(self.s_values)

    def __getitem__(self, idx):
        if self.targets is not None:
            return self.s_values[idx], self.star_info[idx], self.targets[idx]
        else:
            return self.s_values[idx], self.star_info[idx]


class DeepSValuesCNNWithStarInfoModel:
    def __init__(
        self,
        weights_path: Path,
        num_channels: int = 6,
        wavelengths: int = 283,
        num_star_features: int = 3,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        device: torch.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        ),
    ):
        self.weights_path = weights_path
        self.device = device
        self.model = DeepSValuesCNNWithStarInfoTorch(
            num_channels=num_channels,
            wavelengths=wavelengths,
            num_star_features=num_star_features,
        ).to(self.device)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.wavelengths = wavelengths

    def _create_dataloader(
        self,
        s_values: np.ndarray,
        star_info: np.ndarray,
        targets: np.ndarray = None,
        shuffle: bool = True,
    ):
        dataset = _SValuesDataset(s_values, star_info, targets)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _rmse(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        return torch.sqrt(torch.mean((preds - targets) ** 2)).item()

    def train(
        self,
        train_s_values: np.ndarray,
        train_star_info: np.ndarray,
        train_targets: np.ndarray,
        val_s_values: np.ndarray,
        val_star_info: np.ndarray,
        val_targets: np.ndarray,
        epochs: int = 300,
        force_training: bool = False,
    ) -> tuple[float, float, np.ndarray]:
        if os.path.exists(self.weights_path) and not force_training:
            print(f"Weights found at {self.weights_path}. Skipping training.")
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            val_preds = self.predict(val_s_values, val_star_info)
            val_loss = self._rmse(torch.tensor(val_preds), torch.tensor(val_targets))
            train_loss = None
            return val_loss, train_loss, val_preds

        train_loader = self._create_dataloader(train_s_values, train_star_info, train_targets)
        val_loader = self._create_dataloader(
            val_s_values, val_star_info, val_targets, shuffle=False
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=10, factor=0.5
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        train_losses, val_losses = [], []

        progress_bar = tqdm(range(epochs), desc="Training", leave=True)
        for epoch in progress_bar:
            self.model.train()
            running_train_loss = 0.0
            for s_vals, star_inf, targets in train_loader:
                s_vals = s_vals.to(self.device)
                star_inf = star_inf.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(s_vals, star_inf)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * s_vals.size(0)
            train_loss = running_train_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            self.model.eval()
            running_val_loss = 0.0
            val_preds = []
            for s_vals, star_inf, targets in val_loader:
                s_vals = s_vals.to(self.device)
                star_inf = star_inf.to(self.device)
                targets = targets.to(self.device)
                with torch.no_grad():
                    outputs = self.model(s_vals, star_inf)
                val_preds.append(outputs.cpu())
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * s_vals.size(0)
            val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            val_preds = torch.cat(val_preds, dim=0).numpy()
            rmse_val = self._rmse(torch.tensor(val_preds), torch.tensor(val_targets))
            progress_bar.set_description(f"Epoch {epoch + 1}/{epochs}")
            progress_bar.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "val_rmse": f"{rmse_val:.4f}",
                }
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.weights_path)

        # Load best weights
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        val_preds = self.predict(val_s_values, val_star_info)
        final_val_loss = self._rmse(torch.tensor(val_preds), torch.tensor(val_targets))
        final_train_loss = train_losses[-1] if train_losses else None
        return final_val_loss, final_train_loss, val_preds

    def predict(self, s_values: np.ndarray, star_info: np.ndarray) -> np.ndarray:
        self.model.eval()
        if os.path.exists(self.weights_path):
            print(f"Loading weights from {self.weights_path}")
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        else:
            raise FileNotFoundError(
                f"Weights file not found at {self.weights_path}. Please train the model first."
            )
        loader = self._create_dataloader(s_values, star_info, targets=None, shuffle=False)
        preds = []
        with torch.no_grad():
            for s_vals, star_inf in loader:
                s_vals = s_vals.to(self.device)
                star_inf = star_inf.to(self.device)
                outputs = self.model(s_vals, star_inf)
                preds.append(outputs.cpu())
        preds = torch.cat(preds, dim=0).numpy()
        return preds  # (num_samples, wavelengths)


class DeepSValuesCNNWithStarInfoTrainer:
    def __init__(
        self,
        weights_dir: Path,
        n_splits: int = 5,
        num_channels: int = 6,
        wavelengths: int = 283,
        num_star_features: int = 3,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        device: torch.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        ),
    ):
        self.n_splits = n_splits
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.model_params = dict(
            num_channels=num_channels,
            wavelengths=wavelengths,
            num_star_features=num_star_features,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device=device,
        )
        self.models = []

    def train(
        self,
        s_values: np.ndarray,
        star_info: np.ndarray,
        targets: np.ndarray,
        epochs: int = 200,
        force_training: bool = False,
    ) -> tuple[list[float], list[float], np.ndarray]:
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        val_losses, train_losses = [], []
        num_planets = s_values.shape[0]
        wavelengths = self.model_params["wavelengths"]
        val_preds_full = np.zeros((num_planets, wavelengths), dtype=np.float32)
        self.models = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(s_values)):
            print(f"Fold {fold + 1}/{self.n_splits}")
            train_s, val_s = s_values[train_idx], s_values[val_idx]
            train_star, val_star = star_info[train_idx], star_info[val_idx]
            train_t, val_t = targets[train_idx], targets[val_idx]
            weights_path = self.weights_dir / f"model_fold_{fold + 1}.pt"
            model = DeepSValuesCNNWithStarInfoModel(weights_path, **self.model_params)
            val_loss, train_loss, val_preds = model.train(
                train_s,
                train_star,
                train_t,
                val_s,
                val_star,
                val_t,
                epochs=epochs,
                force_training=force_training,
            )
            val_losses.append(val_loss)
            train_losses.append(train_loss)
            val_preds_full[val_idx] = val_preds
            self.models.append(model)
        return val_losses, train_losses, val_preds_full

    def predict(self, s_values: np.ndarray, star_info: np.ndarray) -> np.ndarray:
        if not self.models:
            # Load models from weights_dir
            self.models = []
            for fold in range(1, self.n_splits + 1):
                weights_path = self.weights_dir / f"model_fold_{fold}.pt"
                model = DeepSValuesCNNWithStarInfoModel(weights_path, **self.model_params)
                self.models.append(model)
        preds = [model.predict(s_values, star_info) for model in self.models]
        preds = np.stack(preds, axis=0)  # (n_splits, num_samples, wavelengths)
        return np.mean(preds, axis=0)  # (num_samples, wavelengths)
