import glob
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from ariel_pred.metrics import ariel_score, gll, prmse


class SergeiOldCNN(nn.Module):
    def __init__(self):
        super(SergeiOldCNN, self).__init__()

        self.spectra = nn.Sequential(
            nn.Conv1d(9, 256, 3, padding="same", bias=False),
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

        self.sigma = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        self._to_linear = None
        self._get_conv_output((1, 283))

        self.sigma_out = nn.Linear(self._to_linear, 1)  # type: ignore

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output = self.sigma(input)
        self._to_linear = int(torch.numel(output) / batch_size)

    def forward(self, x_in):
        x = self.spectra(x_in[:, :9])
        spectrum = torch.flatten(x, start_dim=1)
        # y = torch.cat([x_in[:,7:], x-x_in[:,:1,:]], dim=1)
        y = x - x_in[:, :1, :]
        sigma = self.sigma_out(self.sigma(y))

        return spectrum, sigma


class SegeiOldCNNTrainer:
    """Expect data to be of shape (n_samples, 9, 283)"""

    def __init__(self, device: torch.device):
        self.device = device

    def _create_model(self):
        model = SergeiOldCNN().to(self.device)
        return model

    def train(
        self,
        data,
        labels,
        models_save_path,
        epochs=200,
        n_splits=5,
        batch_size=32,
        learning_rate=0.0001,
    ):
        if not os.path.exists(models_save_path):
            os.makedirs(models_save_path)

        kf = KFold(n_splits=n_splits)
        X = list(range(len(data)))

        l2loss = nn.MSELoss()

        progress = tqdm(total=epochs * n_splits, desc="Training SergeiOldCNN")

        for fold, (train_index, val_index) in enumerate(kf.split(X)):  # type: ignore
            print(f"Fold {fold + 1}/{n_splits}")
            model = self._create_model()
            train_x = torch.from_numpy(data[train_index]).float()
            train_y = torch.from_numpy(labels[train_index]).float()
            train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )

            val_x = torch.from_numpy(data[val_index]).float()
            val_y = torch.from_numpy(labels[val_index]).float()
            val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )

            total_train_losses = []

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 200, eta_min=0, last_epoch=-1
            )
            for epoch in range(epochs):
                ep_losses = []
                model.train()
                for i, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    optimizer.zero_grad()
                    spectrum, sigma = model(inputs)

                    loss1 = l2loss(spectrum, targets)
                    mean_diff = (
                        torch.mean((spectrum - targets) ** 2.0, dim=1, keepdims=True) ** 0.5  # type: ignore
                    )
                    loss2 = F.smooth_l1_loss(sigma, mean_diff)

                    loss = loss1 + loss2 * 1e-3
                    loss.backward()
                    optimizer.step()
                    ep_losses.append(loss.item())
                avg_loss = np.mean(ep_losses)
                total_train_losses.append(avg_loss)
                scheduler.step()
                progress.update(1)

            model.eval()
            running_vloss = 0
            preds = np.zeros((len(val_dataset), 283))
            ss = np.zeros((len(val_dataset), 283))
            v_offset = 0
            with torch.no_grad():
                for i, vdata in enumerate(val_loader):
                    vinputs, vtargets = vdata
                    vinputs, vtargets = vinputs.to(self.device), vtargets.to(self.device)
                    vspectrum, vsigma = model(vinputs)
                    preds[v_offset : v_offset + len(vinputs)] = (
                        vspectrum.detach().cpu().numpy() * 1e-3 + 0.0025
                    )
                    ss[v_offset : v_offset + len(vinputs)] = (
                        vsigma.detach().cpu().numpy().clip(0) * 1e-3
                    )
                    vloss = l2loss(vspectrum, vtargets)
                    running_vloss += vloss
                    v_offset += len(vinputs)
            avg_vloss = running_vloss / (i + 1)  # type: ignore
            metric1 = prmse(labels[val_index], preds)
            metric = ariel_score(
                labels[val_index],
                np.concatenate((preds.clip(0), ss.clip(0)), axis=1),
                labels.mean(),
                labels.std(),
                sigma_true=1e-5,
            )

            print(
                "fold {} train {} valid {} rmse {} ariel {}".format(
                    fold + 1,
                    round(avg_loss, 6),  # type: ignore
                    round(avg_vloss.item(), 6),  # type: ignore
                    round(metric1, 6),
                    round(metric, 6),
                )
            )

            torch.save(
                model.state_dict(),
                f"{models_save_path}/sergei_old_cnn_fold{fold + 1}.pth",
            )


class SergeiOldInference:
    """Expect data to be of shape (n_samples, 9, 283)"""

    def __init__(self, models_dir: Path, device: torch.device):
        self.device = device
        self.model_files = sorted(models_dir.glob("sergei_old_cnn_fold*.pth"))

    def predict(self, data, batch_size=32):
        data_x = torch.from_numpy(data).float()
        dataset = torch.utils.data.TensorDataset(data_x)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        preds = np.zeros((len(self.model_files), len(dataset), 283))
        ss = np.zeros((len(self.model_files), len(dataset), 283))

        for model_file in self.model_files:
            model = SergeiOldCNN().to(self.device)
            model.load_state_dict(torch.load(model_file, map_location=self.device))
            model.eval()
            offset = 0
            with torch.no_grad():
                for i, data in enumerate(data_loader):
                    inputs = data[0]
                    inputs = inputs.to(self.device)
                    spectrum, sigma = model(inputs)
                    preds[offset : offset + len(inputs)] += (
                        spectrum.detach().cpu().numpy() * 1e-3 + 0.0025
                    )
                    ss[offset : offset + len(inputs)] += (
                        sigma.detach().cpu().numpy().clip(0) * 1e-3
                    )
                    offset += len(inputs)
        preds = preds.mean(axis=0)
        ss = ss.mean(axis=0)
        return preds.clip(0)


class TransitMultiplicationFactorFinder:
    def __init__(self, poly_degree: int = 3, error_degree: int = 1):
        self.poly_degree = poly_degree
        self.error_degree = error_degree

    def _cost_function(
        self, params: tuple[float], signal: np.ndarray, t1: int, t2: int, t3: int, t4: int
    ) -> float:
        s = params[0]
        y = np.concatenate([signal[:t1], signal[t2:t3] * (s + 1.0), signal[t4:]])
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, deg=self.poly_degree)
        poly = np.poly1d(coeffs)
        fitted = poly(x)
        cost = np.mean(np.abs(y - fitted) ** self.error_degree)
        return float(cost)

    def predict(self, signal: np.ndarray, t1: int, t2: int, t3: int, t4: int) -> float:
        assert len(signal.shape) == 1, (
            "Signal must be a 1D array. Average across wavelengths before passing."
        )
        assert 0 <= t1 < t2 < t3 < t4 < len(signal), (
            "t1, t2, t3, t4 must satisfy 0 <= t1 < t2 < t3 < t4 < signal length."
        )
        initial_s = (
            np.mean(np.concatenate([signal[:t1], signal[t4:]])) / np.mean(signal[t2:t3])
        ) - 1.0
        result = minimize(
            self._cost_function,
            x0=[initial_s],
            args=(signal, t1, t2, t3, t4),
            method="Nelder-Mead",
        )
        return result.x[0]


class SValuesCNN(nn.Module):
    def __init__(self, in_channels: int = 9):
        super(SValuesCNN, self).__init__()

        self.spectra = nn.Sequential(
            nn.Conv1d(in_channels, 256, 3, padding="same", bias=False),
            nn.ReLU(),
            nn.Conv1d(256, 256, 5, padding="same", bias=False),
            nn.ReLU(),
            # nn.Conv1d(256, 256, 7, padding="same", bias=False),
            # nn.ReLU(),
            # nn.Conv1d(256, 256, 9, padding="same", bias=False),
            # nn.ReLU(),
            # nn.Conv1d(256, 256, 11, padding="same", bias=False),
            # nn.ReLU(),
            # nn.Conv1d(256, 256, 13, padding="same", bias=False),
            # nn.ReLU(),
            nn.Conv1d(256, 1, 1, padding="same", bias=False),
        )

        self.sigma = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(1),
        )

    def forward(self, x):
        spectra = self.spectra(x)
        sigma = self.sigma(spectra - x[:, :1, :])
        return spectra.squeeze(1), sigma


class SValuesCNNTrainer:
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
        model = SValuesCNN(in_channels=self.in_channels).to(self.device)
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
            f"{self.models_save_path}/s_values_cnn_fold{fold + 1}.pth",
        )

    def _load_saved_models_paths(self) -> list[Path]:
        model_files = glob.glob(f"{self.models_save_path}/s_values_cnn_fold*.pth")
        return sorted([Path(mf) for mf in model_files])

    def _load_model_from_path(self, model_path: Path) -> nn.Module:
        model = self._make_model()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def _check_model_exists(self) -> bool:
        model_files = glob.glob(f"{self.models_save_path}/s_values_cnn_fold*.pth")
        return len(model_files) > 0

    def _clear_models(self):
        model_files = glob.glob(f"{self.models_save_path}/s_values_cnn_fold*.pth")
        for model_file in model_files:
            os.remove(model_file)

    def _calc_loss(self, spectrum, targets, sigma):
        loss1 = nn.MSELoss()
        loss1_value = loss1(spectrum, targets)
        mean_diff = (
            torch.mean((spectrum - targets) ** 2.0, dim=1, keepdims=True) ** 0.5  # type: ignore
        )
        loss2_value = F.smooth_l1_loss(sigma, mean_diff)
        loss_value = loss1_value + loss2_value * 1e-3
        return loss_value

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
        epoch_val_gll = np.zeros((n_splits, epochs))
        epoch_val_loss = np.zeros((n_splits, epochs))
        epoch_train_loss = np.zeros((n_splits, epochs))

        predectited_spectra = np.zeros((len(data), self.num_channels))
        predectited_sigmas = np.zeros((len(data), self.num_channels))

        for fold, (train_index, val_index) in enumerate(splits):
            model = self._make_model()
            train_loader, val_loader = self._create_data_loaders(
                data * self.train_multiplier,
                train_index,
                val_index,
                labels * self.train_multiplier,
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
                    spectrum, sigma = model(inputs)
                    loss = self._calc_loss(spectrum, targets, sigma)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                avg_loss = running_loss / (i + 1)  # type: ignore
                epoch_train_loss[fold, epoch] = avg_loss  # type: ignore
                scheduler.step()
                progress.update(1)
                val_spectras, val_sigmas, avg_val_loss = self._predict_from_single_model(  # type: ignore
                    model, val_loader, calculate_loss=True
                )
                if return_predictions:
                    predectited_spectra[val_index] = val_spectras
                    predectited_sigmas[val_index] = val_sigmas
                epoch_val_loss[fold, epoch] = avg_val_loss
                val_labels = labels[val_index]
                epoch_val_rmse[fold, epoch] = np.sqrt(np.mean((val_spectras - val_labels) ** 2.0))
                epoch_val_gll[fold, epoch] = gll(
                    np.concatenate((val_spectras, val_sigmas), axis=1), val_labels
                )
                progress.set_postfix(
                    {
                        "fold": fold + 1,
                        "epoch": epoch + 1,
                        "loss": avg_loss,  # type: ignore
                        "val_loss": avg_val_loss,  # type: ignore
                        "val_rmse": epoch_val_rmse[fold, epoch],
                        "val_gll": epoch_val_gll[fold, epoch],
                    }
                )
            self._save_model(model, fold)
        progress.close()
        if return_predictions:
            return (
                predectited_spectra,
                predectited_sigmas,
                (epoch_val_rmse, epoch_val_gll, epoch_val_loss, epoch_train_loss),
            )
        return epoch_val_rmse, epoch_val_gll, epoch_val_loss, epoch_train_loss

    def _predict_from_single_model(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        calculate_loss: bool = False,
    ):
        all_spectrum = np.zeros((len(data_loader.dataset), self.num_channels))  # type: ignore
        all_sigma = np.zeros((len(data_loader.dataset), self.num_channels))  # type: ignore
        offset = 0
        running_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inputs = data[0]
                inputs = inputs.to(self.device)
                spectrum, sigma = model(inputs)
                all_spectrum[offset : offset + len(inputs)] = (
                    spectrum.detach().cpu().numpy() / self.train_multiplier
                )
                all_sigma[offset : offset + len(inputs), :] = (
                    sigma.detach().cpu().numpy().flatten() / self.train_multiplier
                ).clip(1e-10)[:, np.newaxis]
                if calculate_loss:
                    targets = data[1].to(self.device)
                    loss = self._calc_loss(spectrum, targets, sigma)
                    running_loss += loss.item()
                offset += len(inputs)
        if calculate_loss:
            avg_loss = running_loss / (i + 1)  # type: ignore
            return all_spectrum, all_sigma, avg_loss
        return all_spectrum, all_sigma

    def predict(self, data: np.ndarray, batch_size: int = 32):
        if not self._check_model_exists():
            raise ValueError(
                f"No saved models found in {self.models_save_path}. Please train the model first."
            )
        model_paths = self._load_saved_models_paths()
        data_x = torch.from_numpy(data).float()
        dataset = torch.utils.data.TensorDataset(data_x)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        preds = np.zeros((len(model_paths), len(dataset), self.num_channels))
        sigma = np.zeros((len(model_paths), len(dataset), self.num_channels))

        for model_idx, model_path in enumerate(model_paths):
            model = self._load_model_from_path(model_path)
            preds[model_idx], sigma[model_idx] = self._predict_from_single_model(  # type: ignore
                model, data_loader
            )

        preds = preds.mean(axis=0)
        sigma = sigma.mean(axis=0)
        return preds, sigma
