import glob
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
import torch
from torch import nn
from tqdm.auto import tqdm


class OnlySpectrumFromSValuesCNNTorch(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
    ):
        super(OnlySpectrumFromSValuesCNNTorch, self).__init__()

        self.spectra = nn.Sequential(
            nn.Conv1d(in_channels, 256, 3, padding="same", bias=False),
            nn.ReLU(),
            nn.Conv1d(256, 256, 5, padding="same", bias=False),
            nn.ReLU(),
            nn.Conv1d(256, 256, 7, padding="same", bias=False),
            nn.ReLU(),
            nn.Conv1d(256, 256, 9, padding="same", bias=False),
            nn.ReLU(),
            nn.Conv1d(256, 256, 11, padding="same", bias=False),
            nn.ReLU(),
            nn.Conv1d(256, 256, 13, padding="same", bias=False),
            nn.ReLU(),
            nn.Conv1d(256, 1, 1, padding="same", bias=False),
        )

    def forward(self, x):
        spectra = self.spectra(x)
        return spectra.squeeze(1)


class OnlySpectrumFromSValuesCNN:
    def __init__(
        self,
        models_save_path: Path,
        device: torch.device,
        in_channels: int = 9,
        train_multiplier: float = 1.0,
        num_channels: int = 283,
    ):
        self.models_save_path = models_save_path
        self.device = device
        self.in_channels = in_channels
        self.train_multiplier = train_multiplier
        self.num_channels = num_channels

    def _make_model(self):
        model = OnlySpectrumFromSValuesCNNTorch(
            in_channels=self.in_channels,
        ).to(self.device)
        return model

    def _create_data_loaders(
        self,
        data: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ):
        train_x = torch.from_numpy(data[train_idx]).float()
        train_y = torch.from_numpy(labels[train_idx]).float()
        train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        val_x = torch.from_numpy(data[val_idx]).float()
        val_y = torch.from_numpy(labels[val_idx]).float()
        val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def _get_optimizer(self, model: nn.Module, learning_rate: float):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        return optimizer

    def _get_scheduler(self, optimizer: torch.optim.Optimizer, epochs: int = 200):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs, eta_min=0, last_epoch=-1
        )
        return scheduler

    def _save_model(self, model: nn.Module, fold: int):
        torch.save(
            model.state_dict(),
            f"{self.models_save_path}/fold_{fold + 1}.pth",
        )

    def _load_saved_models_paths(self) -> list[Path]:
        model_files = glob.glob(f"{self.models_save_path}/fold_*.pth")
        return sorted([Path(mf) for mf in model_files])

    def _load_model_from_path(self, model_path: Path) -> nn.Module:
        model = self._make_model()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def _check_model_exists(self) -> bool:
        model_files = glob.glob(f"{self.models_save_path}/fold_*.pth")
        return len(model_files) > 0

    def _clear_models(self):
        model_files = glob.glob(f"{self.models_save_path}/fold_*.pth")
        for model_file in model_files:
            os.remove(model_file)

    def _calc_loss(self, spectrum, targets):
        criterion = nn.MSELoss()
        return criterion(spectrum, targets)

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        return data * self.train_multiplier

    def _normalize_labels(self, labels: np.ndarray) -> np.ndarray:
        return labels * self.train_multiplier

    def _denormalize_data(self, data: np.ndarray) -> np.ndarray:
        return data / self.train_multiplier

    def train(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        epochs: int = 200,
        n_splits: int | None = 5,
        batch_size: int = 32,
        learning_rate: float = 0.0001,
        force_retrain: bool = False,
        return_predictions: bool = False,
    ):
        if not os.path.exists(self.models_save_path):
            os.makedirs(self.models_save_path)

        if force_retrain:
            print("Force retrain enabled. Clearing existing models.")
            self._clear_models()
        else:
            if self._check_model_exists():
                print(f"Model files already exist in {self.models_save_path}. Skipping training.")
                return
            print("No saved models found in the specified path. Starting training from scratch.")

        if n_splits is None:
            n_splits = 1
            indices = np.arange(len(data))
            split_at = int(0.25 * len(data))
            train_index, val_index = indices[:split_at], indices[split_at:]
            splits = [(train_index, val_index)]
        else:
            X = list(range(len(data)))
            kf = KFold(n_splits=n_splits)
            splits = kf.split(X)  # type: ignore
        progress = tqdm(total=epochs * n_splits, desc="Training Model")

        epoch_val_rmse = np.zeros((n_splits, epochs))
        epoch_val_loss = np.zeros((n_splits, epochs))
        epoch_train_loss = np.zeros((n_splits, epochs))

        predicted_spectra = np.zeros((len(data), self.num_channels))

        for fold, (train_index, val_index) in enumerate(splits):
            model = self._make_model()
            train_loader, val_loader = self._create_data_loaders(
                self._normalize_data(data),
                train_index,
                val_index,
                self._normalize_labels(labels),
                batch_size,
                shuffle=True,
            )
            optimizer = self._get_optimizer(model, learning_rate)
            scheduler = self._get_scheduler(optimizer, epochs=epochs)

            for epoch in range(epochs):
                running_loss = 0.0
                model.train()
                for i, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    spectrum = model(inputs)
                    loss = self._calc_loss(spectrum, targets)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                avg_loss = running_loss / (i + 1)  # type: ignore
                epoch_train_loss[fold, epoch] = avg_loss  # type: ignore
                scheduler.step()
                progress.update(1)
                val_spectras, avg_val_loss = self._predict_from_single_model(  # type: ignore
                    model, val_loader, calculate_loss=True
                )
                if return_predictions and epoch == epochs - 1:
                    predicted_spectra[val_index] = val_spectras
                epoch_val_loss[fold, epoch] = avg_val_loss
                val_labels = labels[val_index]
                epoch_val_rmse[fold, epoch] = np.sqrt(np.mean((val_spectras - val_labels) ** 2.0))
                progress.set_postfix(
                    {
                        "fold": fold + 1,
                        "epoch": epoch + 1,
                        "loss": avg_loss,  # type: ignore
                        "val_loss": avg_val_loss,  # type: ignore
                        "val_rmse": epoch_val_rmse[fold, epoch],
                    }
                )
            self._save_model(model, fold)
        progress.close()
        if return_predictions:
            return (
                predicted_spectra,
                (epoch_val_rmse, epoch_val_loss, epoch_train_loss),
            )
        return epoch_val_rmse, epoch_val_loss, epoch_train_loss

    def _predict_from_single_model(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        calculate_loss: bool = False,
    ) -> tuple[np.ndarray, float | None]:
        predictions = np.zeros((len(data_loader.dataset), self.num_channels))  # type: ignore
        offset = 0
        running_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inputs = data[0]
                inputs = inputs.to(self.device)
                spectrum = model(inputs)
                predictions[offset : offset + len(inputs)] = self._denormalize_data(
                    spectrum.detach().cpu().numpy()
                )
                if calculate_loss:
                    targets = data[1].to(self.device)
                    loss = self._calc_loss(spectrum, targets)
                    running_loss += loss.item()
                offset += len(inputs)
        if calculate_loss:
            avg_loss = running_loss / (i + 1)  # type: ignore
            return predictions, avg_loss
        return predictions, None

    def predict(self, data: np.ndarray, batch_size: int = 32):
        if not self._check_model_exists():
            raise ValueError(
                f"No saved models found in {self.models_save_path}. Please train the model first."
            )
        model_paths = self._load_saved_models_paths()
        data_x = torch.from_numpy(data).float()
        dataset = torch.utils.data.TensorDataset(data_x)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        predictions = np.zeros((len(model_paths), len(dataset), self.num_channels))

        for model_idx, model_path in enumerate(model_paths):
            model = self._load_model_from_path(model_path)
            predictions[model_idx], _ = self._predict_from_single_model(  # type: ignore
                model, data_loader
            )

        predictions = predictions.mean(axis=0)
        return predictions
