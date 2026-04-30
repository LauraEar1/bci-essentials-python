# This file is based on work Copyright (c) 2023 BCI4Kids
# Original source: https://github.com/kirtonBCIlab/bci-essentials-python
# Licensed under the Mozilla Public License 2.0
# Modifications made by Laura M. L. Earl, [2026]

import numpy as np

from .paradigm import Paradigm


class SsvepParadigm(Paradigm):
    """
    SSVEP paradigm.

    TODO Determine how to automatically set the target frequencies in the classifier from here.
    """

    def __init__(
        self,
        filters=[5, 30],
        iterative_training=False,
        live_update=False,
        buffer_time=0.01,
        unity_epoch_leng=1.0,
        unity_epoch_offset=0.5
    ):
        """
        Parameters
        ----------
        filters : list of floats, *optional*
            Filter bands.
            - Default is `[5, 30]`.
        iterative_training : bool, *optional*
            Flag to indicate if the classifier will be updated iteratively.
            - Default is `False`.
        live_update : bool, *optional*
            Flag to indicate if the classifier will be used to provide
            live updates on trial classification.
            - Default is `False`.
        buffer_time : float, *optional*
            Defines the time in seconds after an epoch for which we require EEG data to ensure that all EEG is present in that epoch.
            - Default is `0.01`.
        Added parameters:
        unity_epoch_len : float, *optional*
            Length of the Unity epoch in seconds.
            - Default is `4.0`.
        unity_epoch_offset : float, *optional*
            Offset of the Unity epoch in seconds.
            - Default is `0.5`.
        """
        super().__init__(filters)

        self.live_update = live_update
        self.iterative_training = iterative_training

        self.lowcut = filters[0]
        self.highcut = filters[1]

        if self.live_update:
            self.classify_each_epoch = True
            self.classify_each_trial = False
        else:
            self.classify_each_trial = True
            self.classify_each_epoch = False

        self.buffer_time = buffer_time
        self.unity_epoch_len = 1.0
        self.unity_epoch_offset = 0.5

        self.paradigm_name = "SSVEP"

        # Online windowing defaults (used by controller during online mode)
        self.online_epoch_len = 2.5
        self.online_stride = 0.25
        self.target_freqs = [8.0, 14.0]

    def get_eeg_start_and_end_times(self, markers, timestamps):
        """
        Get the start and end times of the EEG data based on the markers.

        Parameters
        ----------
        markers : list of str
            List of markers.
        timestamps : list of float
            List of timestamps.

        Returns
        -------
        float
            Start time.
        float
            End time.
        """
        if any("," in m for m in markers):
            start_time = timestamps[0] - self.buffer_time
            last_parts = markers[-1].split(",")
            epoch_len = float(last_parts[3]) if len(last_parts) > 3 else 1.0
            epoch_offset = float(last_parts[4]) if len(last_parts) > 4 else 0.0
            end_time = timestamps[-1] + epoch_offset + epoch_len + self.buffer_time
            return start_time, end_time

        # Unity path — request only the window we actually need
        start_time = timestamps[0]
        for m, ts in zip(markers, timestamps):
            if str(m).strip().upper() == "SSVEP TRIAL STARTED":
                start_time = ts

        end_time = start_time + self.unity_epoch_offset + self.unity_epoch_len
        return start_time - self.buffer_time, end_time + self.buffer_time

    def process_markers(self, markers, marker_timestamps, eeg, eeg_timestamps, fsample):
        """
        This takes in the markers and EEG data and processes them into epochs according to the SSVEP paradigm.

        Parameters
        ----------
        markers : list of str
            List of markers.
        marker_timestamps : list of float
            List of timestamps.
        eeg : np.array
            EEG data. Shape is (n_channels, n_samples).
        eeg_timestamps : np.array
            EEG timestamps. Shape is (n_samples).
        fsample : float
            Sampling frequency.

        Returns
        -------
        np.array
            Processed EEG data. Shape is (n_epochs, n_channels, n_samples).
        np.array
            Labels. Shape is (n_epochs).
        """

    # Unity format — no commas in markers
        if not any("," in m for m in markers):
            label = 0
            start_time = marker_timestamps[0]
            for m, ts in zip(markers, marker_timestamps):
                upper = str(m).strip().upper()
                if upper == "SSVEP CUE LEFT":
                    label = 0
                elif upper == "SSVEP CUE RIGHT":
                    label = 1
                if upper == "SSVEP TRIAL STARTED":
                    start_time = ts

            n_channels, _ = eeg.shape
            marker_eeg_timestamps = eeg_timestamps - start_time
            epoch_time = np.arange(
                self.unity_epoch_offset,
                self.unity_epoch_offset + self.unity_epoch_len,
                1 / fsample
            )
            epoch_eeg = np.zeros((1, n_channels, len(epoch_time)))
            for c in range(n_channels):
                epoch_eeg[0, c, :] = np.interp(epoch_time, marker_eeg_timestamps, eeg[c, :])
            epoch_eeg[0, :, :] = super()._preprocess(epoch_eeg[0, :, :], fsample, self.lowcut, self.highcut)
            return epoch_eeg, np.array([label], dtype=int)

        # Legacy format — original code unchanged below
        y = np.zeros(len(markers), dtype=int)
        for i, marker in enumerate(markers):
            marker = marker.split(",")

            # Expected format:
            # SSVEP,TRIAL,{label},{epoch_len},{offset},{freq1},{freq2},...
            if len(marker) < 4:
                raise ValueError(f"Malformed SSVEP marker: {markers[i]}")

            label = int(float(marker[2]))
            epoch_length = float(marker[3])
            epoch_offset = float(marker[4]) if len(marker) > 4 else 0.0

            n_channels, _ = eeg.shape

            marker_timestamp = marker_timestamps[i]

            # Subtract the marker timestamp from the EEG timestamps so that 0 becomes the marker onset
            marker_eeg_timestamps = eeg_timestamps - marker_timestamp

            # Create the epoch time vector
            epoch_time = np.arange(epoch_offset, epoch_offset + epoch_length, 1 / fsample)

            # Initialize the EEG data array
            epoch_eeg = np.zeros((1, n_channels, len(epoch_time)))

            # Interpolate the EEG data to the epoch time vector for each channel
            for c in range(n_channels):
                epoch_eeg[0, c, :] = np.interp(
                    epoch_time, marker_eeg_timestamps, eeg[c, :]
                )

            epoch_eeg[0, :, :] = super()._preprocess(
                epoch_eeg[0, :, :], fsample, self.lowcut, self.highcut
            )

            if i == 0:
                X = epoch_eeg
            else:
                X = np.concatenate((X, epoch_eeg), axis=0)

            y[i] = label
        return X, y

    # TODO: Implement this
    def check_compatibility(self):
        pass
