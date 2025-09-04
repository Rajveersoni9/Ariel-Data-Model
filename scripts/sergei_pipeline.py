import typer

from ariel_pred.config import CalibrationConfig, Config
from ariel_pred.dataset import DataLoaderAndCalibrator

app = typer.Typer()


@app.command()
def main():
    global_config = Config()
    calibration_config = CalibrationConfig(
        data_path=global_config.DATA_PATH,
        binning=1,
        airs_lower_channel=0,
        airs_upper_channel=356,
        preprocessing_n_jobs=4,
    )
    signal_processor = DataLoaderAndCalibrator(cfg=calibration_config)
    train_data = signal_processor.process_all_data("train")
    test_data = signal_processor.process_all_data("test")
    typer.echo(f"Train data shape: {train_data.shape}")
    typer.echo(f"Test data shape: {test_data.shape}")


if __name__ == "__main__":
    app()
