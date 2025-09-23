import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class ResNetAndCNNWithSigmaModelTorch(nn.Module):
    def __init__(
        self,
        num_channels=6,
        wavelengths=283,
        num_star_features=3,
        spectrum_num_resnet_blocks=3,
        spectrum_resnet_hidden=512,
        sigma_hidden=512,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.wavelengths = wavelengths
        self.num_star_features = num_star_features
        # CNN layers for s-values
        self.cnn1 = nn.Conv1d(num_channels, 32, kernel_size=5, padding="same")
        self.cnn2 = nn.Conv1d(32, 16, kernel_size=7, padding="same")
        self.cnn3 = nn.Conv1d(16, 1, kernel_size=9, padding="same")

        # ResNet blocks for spectrum prediction
        self.spectrum_fc = nn.Sequential(
            nn.Linear(wavelengths + num_star_features, spectrum_resnet_hidden),
            nn.ReLU(),
            *[ResNetBlock(spectrum_resnet_hidden) for _ in range(spectrum_num_resnet_blocks)],
            nn.Linear(spectrum_resnet_hidden, wavelengths),
        )

        # Sigma prediction
        self.sigma_fc = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=7, padding="same"),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=9, padding="same"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(wavelengths, sigma_hidden),
            nn.ReLU(),
            nn.Linear(sigma_hidden, 1),
        )

    def forward(self, s_values, star_info):
        # Pass through CNN
        x = F.relu(self.cnn1(s_values))
        x = F.relu(self.cnn2(x))
        x = self.cnn3(x)
        x = x.squeeze(1)

        x_cat = torch.cat([x, star_info], dim=1)

        # Pass through spectrum ResNet
        spectrum_pred = self.spectrum_fc(x_cat)

        # Pass through sigma model
        sigma_pred = self.sigma_fc(spectrum_pred.unsqueeze(1) - s_values[:, :1, :]).squeeze(1)

        return spectrum_pred, sigma_pred


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


class ResNetAndCNNWithSigmaModel:
    def __init__(
        self,
        weights_path: Path,
        num_channels: int = 6,
        wavelengths: int = 283,
        num_star_features: int = 3,
        spectrum_num_resnet_blocks: int = 3,
        spectrum_resnet_hidden: int = 512,
        sigma_hidden: int = 512,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        underperforming_mul: float = 10.0,
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
        self.model = ResNetAndCNNWithSigmaModelTorch(
            num_channels=num_channels,
            wavelengths=wavelengths,
            num_star_features=num_star_features,
            spectrum_num_resnet_blocks=spectrum_num_resnet_blocks,
            spectrum_resnet_hidden=spectrum_resnet_hidden,
            sigma_hidden=sigma_hidden,
        ).to(self.device)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.wavelengths = wavelengths
        self.underperforming_mul = underperforming_mul

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
        epochs: int = 200,
        force_training: bool = False,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        if os.path.exists(self.weights_path) and not force_training:
            print(f"Weights found at {self.weights_path}. Skipping training.")
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            val_spectrum, val_sigma = self.predict(val_s_values, val_star_info)
            val_loss = self._rmse(torch.tensor(val_spectrum), torch.tensor(val_targets))
            train_loss = None
            return val_loss, train_loss, val_spectrum, val_sigma

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
                spectrum_pred, sigma_pred = self.model(s_vals, star_inf)
                mean_diff = torch.mean((spectrum_pred - targets) ** 2, dim=1) ** 0.5
                loss1 = criterion(spectrum_pred, targets)
                loss2 = torch.sqrt(
                    torch.mean(
                        torch.where(
                            sigma_pred >= mean_diff,
                            (sigma_pred - mean_diff) ** 2,
                            (self.underperforming_mul * (sigma_pred - mean_diff)) ** 2,
                        )
                    )
                )
                loss = loss1 + loss2 * 1e-3
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * s_vals.size(0)
            train_loss = running_train_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            self.model.eval()
            running_val_loss = 0.0
            val_spectrum_preds, val_sigma_preds = [], []
            for s_vals, star_inf, targets in val_loader:
                s_vals = s_vals.to(self.device)
                star_inf = star_inf.to(self.device)
                targets = targets.to(self.device)
                with torch.no_grad():
                    spectrum_pred, sigma_pred = self.model(s_vals, star_inf)
                val_spectrum_preds.append(spectrum_pred.cpu())
                val_sigma_preds.append(sigma_pred.cpu())
                mean_diff = torch.mean((spectrum_pred - targets) ** 2, dim=1) ** 0.5
                loss1 = criterion(spectrum_pred, targets)
                loss2 = torch.sqrt(
                    torch.mean(
                        torch.where(
                            sigma_pred >= mean_diff,
                            (sigma_pred - mean_diff) ** 2,
                            (self.underperforming_mul * (sigma_pred - mean_diff)) ** 2,
                        )
                    )
                )
                batch_loss = loss1 + loss2 * 1e-3
                running_val_loss += batch_loss.item() * s_vals.size(0)
            val_loss = running_val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            val_spectrum_preds = torch.cat(val_spectrum_preds, dim=0).numpy()
            val_sigma_preds = torch.cat(val_sigma_preds, dim=0).numpy()
            rmse_val = self._rmse(torch.tensor(val_spectrum_preds), torch.tensor(val_targets))
            rmse_sigma = self._rmse(
                torch.tensor(val_sigma_preds),
                torch.tensor(np.abs(val_spectrum_preds - val_targets).mean(axis=1)),
            )
            progress_bar.set_description(f"Epoch {epoch + 1}/{epochs}")
            progress_bar.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "val_rmse": f"{rmse_val:.4f}",
                    "val_rmse_sigma": f"{rmse_sigma:.4f}",
                }
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.weights_path)

        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        val_spectrum, val_sigma = self.predict(val_s_values, val_star_info)
        final_val_loss = self._rmse(torch.tensor(val_spectrum), torch.tensor(val_targets))
        final_train_loss = train_losses[-1] if train_losses else None
        return final_val_loss, final_train_loss, val_spectrum, val_sigma

    def predict(
        self, s_values: np.ndarray, star_info: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        if os.path.exists(self.weights_path):
            print(f"Loading weights from {self.weights_path}")
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        else:
            raise FileNotFoundError(
                f"Weights file not found at {self.weights_path}. Please train the model first."
            )
        loader = self._create_dataloader(s_values, star_info, targets=None, shuffle=False)
        spectrum_preds, sigma_preds = [], []
        with torch.no_grad():
            for s_vals, star_inf in loader:
                s_vals = s_vals.to(self.device)
                star_inf = star_inf.to(self.device)
                spectrum_pred, sigma_pred = self.model(s_vals, star_inf)
                spectrum_preds.append(spectrum_pred.cpu())
                sigma_preds.append(sigma_pred.cpu())
        spectrum_preds = torch.cat(spectrum_preds, dim=0).numpy()
        sigma_preds = torch.cat(sigma_preds, dim=0).numpy()
        return spectrum_preds, sigma_preds


class ResNetAndCNNWithSigmaModelTrainer:
    def __init__(
        self,
        weights_dir: Path,
        n_splits: int = 5,
        num_channels: int = 6,
        wavelengths: int = 283,
        num_star_features: int = 3,
        spectrum_num_resnet_blocks: int = 3,
        spectrum_resnet_hidden: int = 512,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        underperforming_mul: float = 10.0,
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
            spectrum_num_resnet_blocks=spectrum_num_resnet_blocks,
            spectrum_resnet_hidden=spectrum_resnet_hidden,
            learning_rate=learning_rate,
            batch_size=batch_size,
            underperforming_mul=underperforming_mul,
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
    ) -> tuple[list[float], list[float], np.ndarray, np.ndarray]:
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        val_losses, train_losses = [], []
        num_planets = s_values.shape[0]
        wavelengths = self.model_params["wavelengths"]
        val_spectrum_full = np.zeros((num_planets, wavelengths), dtype=np.float32)
        val_sigma_full = np.zeros((num_planets,), dtype=np.float32)
        self.models = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(s_values)):
            print(f"Fold {fold + 1}/{self.n_splits}")
            train_s, val_s = s_values[train_idx], s_values[val_idx]
            train_star, val_star = star_info[train_idx], star_info[val_idx]
            train_t, val_t = targets[train_idx], targets[val_idx]
            weights_path = self.weights_dir / f"model_fold_{fold + 1}.pt"
            model = ResNetAndCNNWithSigmaModel(weights_path, **self.model_params)
            val_loss, train_loss, val_spectrum, val_sigma = model.train(
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
            val_spectrum_full[val_idx] = val_spectrum
            val_sigma_full[val_idx] = val_sigma
            self.models.append(model)
        return val_losses, train_losses, val_spectrum_full, val_sigma_full

    def predict(
        self, s_values: np.ndarray, star_info: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.models:
            # Load models from weights_dir
            self.models = []
            for fold in range(1, self.n_splits + 1):
                weights_path = self.weights_dir / f"model_fold_{fold}.pt"
                model = ResNetAndCNNWithSigmaModel(weights_path, **self.model_params)
                self.models.append(model)
        spectrum_preds = []
        sigma_preds = []
        for model in self.models:
            spectrum_pred, sigma_pred = model.predict(s_values, star_info)
            spectrum_preds.append(spectrum_pred)
            sigma_preds.append(sigma_pred)
        spectrum_preds = np.stack(spectrum_preds, axis=0)  # (n_splits, num_samples, wavelengths)
        sigma_preds = np.stack(sigma_preds, axis=0)  # (n_splits, num_samples)
        return np.mean(spectrum_preds, axis=0), np.mean(sigma_preds, axis=0)
