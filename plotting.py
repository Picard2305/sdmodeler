from pathlib import Path

from sdmodeler.identification.fitting import FittedModel
from sdmodeler.validation import ValidationResult


def save_validation_plot(model: FittedModel, validation: ValidationResult) -> None:
    if not validation.applicable:
        return

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 5))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    for index, variable in enumerate(model.x_variables):
        axes.plot(validation.timestamps, validation.actual_trajectory[:, index], label=f"{variable} observed")
        axes.plot(validation.timestamps, validation.predicted_trajectory[:, index], label=f"{variable} simulated")
    axes.set_title("Observed vs simulated dynamics (validation window)")
    axes.set_xlabel("Time")
    axes.set_ylabel("State")
    axes.legend()
    figure.autofmt_xdate()
    output_path = Path("ise_robustness_threshold.png")
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved observed-versus-simulated plot to {output_path}")
