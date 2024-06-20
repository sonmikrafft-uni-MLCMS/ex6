# module for plotting

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.cm as cm
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from tqdm import tqdm


def simple_histogram(data: np.ndarray, title: str, xlabel: str, legend_label: str, num_bins: int = 10, ax: plt.Axes = None) -> plt.Axes:
    """Create a simple histogram plot.

    Args:
        data (np.ndarray): The data to plot.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        legend_label: The label for the plot presented in the legend
        ylabel (str): The label for the y-axis.
        num_bins (int, optional): The number of bins to use. Defaults to 10.
        ax (plt.Axes, optional): The axes to plot on. Defaults to None.
    """

    if ax is None:
        _, ax = plt.subplots()

    ax.hist(data, num_bins, label=legend_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    ax.grid(True)

    return ax


def simple_scatter(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    xlim: list = None,
    ylim: list = None,
    axis_equal: bool = True,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Create a simple scatter plot.

    Args:
        x (np.ndarray): The x-data to plot.
        y (np.ndarray): The y-data to plot.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        xlim (list, optional): The x-axis limits. Defaults to None.
        ylim (list, optional): The y-axis limits. Defaults to None.
        axis_equal (bool, optional): Whether to set the axes equal. Defaults to True.
        ax (plt.Axes, optional): The axes to plot on. Defaults to None.
        **kwargs: Additional keyword arguments to pass to plt.scatter.
    """

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(x, y, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)

    if axis_equal:
        ax.axis("equal")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return ax


def prediction_scatter(y_pred: np.ndarray, y_true: np.ndarray, title: str, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    """Create a scatter plot of the prediction vs the true value.

    Args:
        y_pred (np.ndarray): The predicted values.
        y_true (np.ndarray): The true values.
        title (str): The title of the plot.
        ax (plt.Axes, optional): The axes to plot on. Defaults to None.
        **kwargs: Additional keyword arguments to pass to plt.scatter.
    """

    if ax is None:
        _, ax = plt.subplots()

    ax = simple_scatter(
        x=y_true,
        y=y_pred,
        title=title,
        xlabel="True",
        ylabel="Predicted",
        ax=ax,
        label="predictions",
        axis_equal=False,
        **kwargs,
    )

    y_true_min = np.min(y_true)
    y_true_max = np.max(y_true)
    ax.plot([y_true_min, y_true_max], [y_true_min, y_true_max], "r--", label="perfect prediction")

    # ax.legend()

    return ax


def create_walking_video(df: pd.DataFrame, filename: str, first_n: int = 500, interval: float = 1000 / 16) -> None:
    """Create a video of the walking data.

    Args:
        df (pd.DataFrame): The data to plot.
        filename (str): The filename to save the video to.
        first_n (int, optional): Only show the first n frames. Defaults to None.
        interval (float, optional): The interval between frames in milliseconds. Defaults to 1000 / 16.
    """
    x_min, x_max = df["X"].min(), df["X"].max()
    y_min, y_max = df["Y"].min(), df["Y"].max()

    fig_size = (2, 5)

    frame_names = df["FRAME"].unique()
    frame_names = frame_names[: min(first_n, len(frame_names))] if first_n is not None else frame_names
    imgs = []
    print("Creating images...")
    for frame in tqdm(frame_names, total=len(frame_names)):
        df_frame = df[df["FRAME"] == frame]

        fig, ax = plt.subplots(figsize=fig_size)
        width, height = fig.get_size_inches() * fig.get_dpi()
        canvas = FigureCanvas(fig)
        ax = simple_scatter(
            df_frame["X"],
            df_frame["Y"],
            title="",
            xlabel="X [m]",
            ylabel="Y [m]",
            xlim=[x_min, x_max],
            ylim=[y_min, y_max],
            ax=ax,
        )
        plt.gca().set_aspect("equal", adjustable="box")

        canvas.draw()  # draw the canvas, cache the renderer
        img = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(int(height), int(width), 3)

        imgs.append(img)
        plt.close()

    frames = []  # for storing the generated images
    fig, ax = plt.subplots(figsize=fig_size)
    print("Creating video frames...")
    for image in tqdm(imgs, total=len(imgs)):
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        frames.append([ax.imshow(image, cmap=cm.Greys_r, animated=True)])

    print("Creating animation...")
    ani = animation.ArtistAnimation(fig, frames, interval=interval, repeat=False)
    ani.save(filename)
    plt.close()


def frame_evolution_plot(
    df: pd.DataFrame, start_frame: int, num_frames: int, delta_frames: int
) -> tuple[plt.Figure, plt.Axes]:
    """Shows the pedestrian positions as a scatter plot over several consecutive frames.

    Args:
        df (pd.DataFrame): The data to plot.
        start_frame (int): The frame to start at.
        num_frames (int): The number of frames to show.
        delta_frames (int): The number of frames to skip between each frame.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes of the plot.
    """
    cm.get_cmap("hsv", len(df["ID"].unique()))
    colors = cm.get_cmap("tab20", len(df["ID"].unique())).colors
    colors_dict = {unique_id: colors[np.random.randint(low=0, high=100, size=1)] for unique_id in df["ID"].unique()}

    frames = np.linspace(start_frame, start_frame + (num_frames - 1) * delta_frames, num=num_frames, dtype=int)
    frame_suffix = df["FRAME"][0].split("_")[-1]

    x_min, x_max = df["X"].min(), df["X"].max()
    y_min, y_max = df["Y"].min(), df["Y"].max()

    fig, axs = plt.subplots(1, len(frames), figsize=(2 * len(frames), 5))
    for i, frame in enumerate(frames):
        frame_name = f"{frame}_{frame_suffix}"
        df_frame = df[df["FRAME"] == frame_name]
        colors = [colors_dict[unique_id] for unique_id in df_frame["ID"]]
        ax = axs[i]
        ax = simple_scatter(
            df_frame["X"],
            df_frame["Y"],
            title=f"Frame {frame}",
            xlabel="X [m]",
            ylabel="Y [m]",
            xlim=[x_min, x_max],
            ylim=[y_min, y_max],
            ax=ax,
            c=colors,
        )
    fig.tight_layout()

    return fig, ax


def plot_trajectories(df: pd.DataFrame, ids: list[int], title: str) -> tuple[plt.Figure, plt.Axes]:
    """Plots the trajectories of the given pedestrians.

    Args:
        df (pd.DataFrame): The data to plot.
        ids (list[int]): The IDs of the pedestrians to plot.
        title (str): The title of the plot.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes of the plot.
    """
    cm.get_cmap("hsv", len(df["ID"].unique()))
    colors = cm.get_cmap("tab20", len(df["ID"].unique())).colors
    colors_dict = {unique_id: colors[np.random.randint(low=0, high=100, size=1)] for unique_id in df["ID"].unique()}

    x_min, x_max = df["X"].min(), df["X"].max()
    y_min, y_max = df["Y"].min(), df["Y"].max()

    suffix = df["ID"][0].split("_")[-1]
    IDs = [f"{id}_{suffix}" for id in ids]

    fig, ax = plt.subplots(1, 1, figsize=(3, 5))
    df_id = df[df["ID"].isin(IDs)]
    colors = [colors_dict[unique_id] for unique_id in df_id["ID"]]

    for id in IDs:
        df_single_id = df_id[df_id["ID"] == id]
        ax = simple_scatter(
            df_single_id["X"],
            df_single_id["Y"],
            title=title,
            xlabel="X [m]",
            ylabel="Y [m]",
            xlim=[x_min, x_max],
            ylim=[y_min, y_max],
            ax=ax,
            s=2,
            label=id.split("_")[0],
        )
    # plt.gca().set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    # fig.tight_layout()

    return fig, ax


def plot_2d_tuning_plot(
    name_hyperparameter: str,
    name_metric: str,
    name_model: str,
    hyperparams: np.ndarray,
    mean_score_train: float,
    mean_score_val: float,
    std_score_train: float = None,
    std_score_val: float = None,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Plots tuning plot that shows the mean and standard deviation of the given metric for each hyperparameter on
    both the training and validation set.

    Args:
        name_hyperparameter (str): The name of the hyperparameter. This is used as the x-axis label.
        name_metric (str): The name of the metric. This is used as the y-axis label.
        name_model (str): The name of the model. This is used as the title.
        hyperparams (np.ndarray): The hyperparameters to plot. Of shape (num_hyperparams,).
        mean_score_train (float): The mean score on the training set. Of shape (num_hyperparams,).
        mean_score_val (float): The mean score on the validation set. Of shape (num_hyperparams,).
        std_score_train (float, optional): The standard deviation of the score on the training set. Defaults to None.
            Of shape (num_hyperparams,).
        std_score_val (float, optional): The standard deviation of the score on the validation set. Defaults to None.
            Of shape (num_hyperparams,).
        ax (plt.Axes, optional): The axes to plot on. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the plot function.

    Returns:
        plt.Axes: The axes of the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.plot(hyperparams, mean_score_train, label="train", **kwargs)
    if std_score_train is not None:
        ax.fill_between(hyperparams, mean_score_train - std_score_train, mean_score_train + std_score_train, alpha=0.2)
    ax.plot(hyperparams, mean_score_val, label="val", **kwargs)
    if std_score_val is not None:
        ax.fill_between(hyperparams, mean_score_val - std_score_val, mean_score_val + std_score_val, alpha=0.2)

    ax.set_xlabel(name_hyperparameter)
    ax.set_ylabel(name_metric)
    ax.set_title(f"{name_model} - Hyperparameter Tuning")

    ax.grid()
    ax.legend()

    return ax
