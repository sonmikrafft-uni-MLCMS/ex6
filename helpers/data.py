# module to load data
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import pickle
from glob import glob

tqdm.pandas()


class SpeedExperimentDataset:
    def __init__(self, data_dir: str) -> None:
        # data
        self.data_dir = data_dir
        self.data_files = glob(os.path.join("*", "*", data_dir, "*.txt"))
        self.data_files.sort()
        self.df = pd.DataFrame()

        # preparation
        self.col_names = ["ID", "FRAME", "X", "Y", "Z"]
        self.distance_factor_to_m = 1 / 100
        self.time_delta_s = 1 / 16

        # states
        self.data_loaded = False
        self.vel_calculated = False
        self.appended_with_k_nearest_neighbors = False

    def __repr__(self) -> str:
        return self.df.__repr__()

    @staticmethod
    def load(save_dir: str, filename: str) -> "SpeedExperimentDataset":
        """Load the pickled object instance and return it.

        Args:
            save_dir (str): The directory to load the object from.
            filename (str): Filename.

        Raises:
            FileNotFoundError: If the directory does not exist.

        Returns:
            SpeedExperimentDataset: The loaded object.
        """
        filehandler = open(os.path.join(save_dir, f"{filename}.pkl"), "rb")
        obj = pickle.load(filehandler)
        filehandler.close()
        return obj

    def save(self, save_dir: str, filename: str) -> None:
        """Pickle the object.

        Args:
            save_dir (str): The directory to save the object to.
            filename (str): Filename.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        filehandler = open(os.path.join(save_dir, f"{filename}.pkl"), "wb")
        pickle.dump(self, filehandler)
        filehandler.close()

    def save_data(self, save_dir: str, filename: str) -> None:
        """Save the data only.

        Args:
            save_dir (str): The directory to save the data to.
            filename (str): The filename.
        """
        self.df.to_csv(os.path.join(save_dir, f"{filename}.csv"), index=False)

    def read_data(self, only_n_files: int = None, force: bool = False) -> "SpeedExperimentDataset":
        """Read either the raw bottleneck or the corridor data.

        Args:
            only_n_files (int, optional): Only load the first n files. Defaults to None. Then all files are loaded.
            force (bool, optional): Force the loading of the data. Defaults to False.

        Raises:
            ValueError: If category is not either "bottleneck" or "corridor".

        Returns:
            SpeedExperimentDataset: self
        """
        if (not force) and self.data_loaded:
            print("Data already loaded. Use force=True to force loading.")
            return self

        # reset states
        self.vel_calculated = False
        self.appended_with_k_nearest_neighbors = False

        # load data
        df_list = []
        max_num = len(self.data_files) if only_n_files is None else min(only_n_files, len(self.data_files))
        print(f"Loading {max_num} files ...")
        for filename in tqdm(self.data_files[:max_num], total=max_num):
            df = pd.read_csv(filename, index_col=None, header=None, names=self.col_names, delimiter=" ")
            df["ORIGINAL_FILENAME"] = filename.split("/")[-1].split(".")[0]
            df_list.append(df)
        df_full = pd.concat(df_list, axis=0, ignore_index=True)

        # make ID and FRAME unique across all files
        print("Making ID and FRAME unique across all files...")
        self.df = _make_unique_across_files(df_full)

        # fix position units from cm to m
        print("Fixing position units from cm to m...")
        self.df[["X", "Y", "Z"]] *= self.distance_factor_to_m
        self.distance_factor_to_m = 1

        # set state
        self.data_loaded = True

        return self

    def calculate_velocity(self, force: bool = False) -> "SpeedExperimentDataset":
        """Calculate the velocity of the pedestrians by using the previous and next position values.

        Args:
            df (pd.DataFrame): The data to calculate the velocity for.
            force (bool, optional): Force the calculation of the velocity. Defaults to False.

        Returns:
            SpeedExperimentDataset: self
        """
        if (not force) and self.vel_calculated:
            print("Velocity already calculated. Use force=True to force calculation.")
            return self

        print("Calculating velocities...")

        df = self.df.copy()

        # iterate over all unique pedestrians and for each add previous and next position values
        print("Step 1/2: Adding previous and next position values...")
        df_new_list = []
        unique_pedestrians = df["ID"].unique()
        for unique_id in tqdm(unique_pedestrians, total=len(unique_pedestrians)):
            df_sub = df[df["ID"] == unique_id].copy()

            # add x, y, z value of next row to each row
            df_sub["X_NEXT"] = df_sub["X"].shift(-1)
            df_sub["Y_NEXT"] = df_sub["Y"].shift(-1)
            df_sub["Z_NEXT"] = df_sub["Z"].shift(-1)

            # add x, y, z value of previous row to each row
            df_sub["X_PREV"] = df_sub["X"].shift(1)
            df_sub["Y_PREV"] = df_sub["Y"].shift(1)
            df_sub["Z_PREV"] = df_sub["Z"].shift(1)

            df_new_list.append(df_sub)
        df_new = pd.concat(df_new_list, axis=0, ignore_index=True)

        # calculate velocity
        print("Step 2/2: Calculating velocities...")
        df_new["VEL"] = df_new.progress_apply(
            _calc_v, axis=1, delta_t_s=self.time_delta_s, distance_factor_to_m=self.distance_factor_to_m
        )

        # drop unnecessary columns
        df_new = df_new.drop(columns=["X_NEXT", "Y_NEXT", "Z_NEXT", "X_PREV", "Y_PREV", "Z_PREV"])
        self.df = df_new

        # set state
        self.vel_calculated = True

        return self

    def append_with_k_nearest_neighbors(self, k: int, force: bool = False) -> "SpeedExperimentDataset":
        """For each pedestrian at each frame, find the k nearest pedestrians at the same frame and append data about them
        to the current row as new columns.

        Deltas are calculated as position of current predestrian - other pedestrian.

        Columns are added:
            - AVG_DISTANCE_TO_K_CLOSEST: The average distance to the k closest pedestrians at the same frame.
            - DX_1, ..., DX_K: The x-distance to the k closest pedestrians at the same frame.
            - DY_1, ..., DY_K: The y-distance to the k closest pedestrians at the same frame.
            - DISTS_1, ..., DISTS_K: The distance to the k closest pedestrians at the same frame.
            - ID_1, ..., ID_K: The ID of the k closest pedestrians at the same frame.

        Args:
            df (pd.DataFrame): The data to append the k nearest pedestrians to.
            k (int): The number of nearest pedestrians to append.
            force (bool, optional): Force the calculation of the k nearest neighbors. Defaults to False.

        Returns:
            SpeedExperimentDataset: self
        """
        if (not force) and self.appended_with_k_nearest_neighbors:
            print("K nearest neighbors already appended. Use force=True to force appending.")
            return self

        df = self.df.copy()

        print("Appending k nearest pedestrians to dataframe. This might take a while...")
        for this_index, this_row in tqdm(df.iterrows(), total=df.shape[0]):
            this_unique_frame = this_row["FRAME"]
            this_unique_id = this_row["ID"]

            # find pedestrians in the same frame in the same file, other than this one
            same_frame = df[df["FRAME"] == this_unique_frame]
            same_frame = same_frame[same_frame["ID"] != this_unique_id]

            # calculate distance to all pedestrians in the same frame
            factor_to_m = 1 / 100
            delta_x = (this_row["X"] - same_frame["X"]) * factor_to_m
            delta_y = (this_row["Y"] - same_frame["Y"]) * factor_to_m
            delta_z = (this_row["Z"] - same_frame["Z"]) * factor_to_m
            distances = ((delta_x) ** 2 + (delta_y) ** 2 + (delta_z) ** 2) ** 0.5

            # sort by distance descending and take first 10
            closest_distances = distances.sort_values(ascending=True)[:k]
            avg_closest_distances = closest_distances.mean()

            # index and ids
            closest_indices = closest_distances.index
            closest_ids = same_frame.loc[closest_indices, "ID"]

            # delta x, y
            closest_distances_x = delta_x.loc[closest_indices]
            closest_distances_y = delta_y.loc[closest_indices]

            # add all of this information as new columns
            df.loc[this_index, "AVG_DISTANCE_TO_K_CLOSEST"] = avg_closest_distances
            for i in range(len(closest_indices)):
                column_str_x = f"DX_{i+1}"  # DX_1, DX_2, DX_3, ...
                column_str_y = f"DY_{i+1}"  # DY_1, DY_2, DY_3, ...
                column_str_dist = f"DIST_{i+1}"  # DIST_1, DIST_2, DIST_3, ...
                column_str_id = f"ID_{i+1}"  # ID_1, ID_2, ID_3, ...
                df.loc[this_index, column_str_id] = closest_ids.iloc[i]
                df.loc[this_index, column_str_dist] = closest_distances.iloc[i]
                df.loc[this_index, column_str_x] = closest_distances_x.iloc[i]
                df.loc[this_index, column_str_y] = closest_distances_y.iloc[i]

        self.df = df

        # set state
        self.k_nearest_neighbors_appended = True

        return self


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- HELPER FUNCTIONS --------------------------------------------------


def _calc_v(row: pd.Series, delta_t_s: float, distance_factor_to_m: float) -> float:
    """Calculate the velocity of one pedestrian by using the previous and next position values.

    Assumes the data to have the following columns in the same row:
        - X: x-coordinate of the current position
        - Y: y-coordinate of the current position
        - Z: z-coordinate of the current position
        - X_PREV: x-value of previous position
        - Y_PREV: y-value of previous position
        - Z_PREV: z-value of previous position
        - X_NEXT: x-value of next position
        - Y_NEXT: y-value of next position
        - Z_NEXT: z-value of next position

    Args:
        row (pd.Series): The row to calculate the velocity for.
        delta_t_s (float): The time difference between one and a next position. In seconds.
        distance_factor_to_m (float): The factor to convert the distance to meters.

    Returns:
        float: The velocity of the pedestrian in m/s.
    """
    ds = 0

    if np.isnan(row["X_PREV"]):  # first row, use this position and next position
        ds = (
            (row["X_NEXT"] - row["X"]) ** 2 + (row["Y_NEXT"] - row["Y"]) ** 2 + (row["Z_NEXT"] - row["Z"]) ** 2
        ) ** 0.5
    elif np.isnan(row["X_NEXT"]):  # last row, use previous position and this position
        ds = (
            (row["X"] - row["X_PREV"]) ** 2 + (row["Y"] - row["Y_PREV"]) ** 2 + (row["Z"] - row["Z_PREV"]) ** 2
        ) ** 0.5
    else:  # use central difference
        ds = (
            (row["X_NEXT"] - row["X_PREV"]) ** 2
            + (row["Y_NEXT"] - row["Y_PREV"]) ** 2
            + (row["Z_NEXT"] - row["Z_PREV"]) ** 2
        ) ** 0.5 / 2

    return ds * distance_factor_to_m / delta_t_s


def _make_unique_across_files(df: pd.DataFrame) -> pd.DataFrame:
    """Make the data unique across all files.

    The issue is that across different files there are the same IDs and frames although they are not the same but from
    different experiments. This function makes the data unique by adding the filename to the ID and frame.

    Changes the values of the following columns:
        - ID: Adds the filename to the ID.
        - FRAME: Adds the filename to the frame.

    Args:
        df (pd.DataFrame): The data to make unique.

    Returns:
        pd.DataFrame: The same data after making the columns unique.
    """
    df = df.copy()
    df["ID"] = df["ID"].astype(str) + "_" + df["ORIGINAL_FILENAME"]
    df["FRAME"] = df["FRAME"].astype(str) + "_" + df["ORIGINAL_FILENAME"]
    return df
