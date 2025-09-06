import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from ariel_pred.metrics import ariel_score, prmse


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
