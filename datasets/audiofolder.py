import os
import numpy as np
import torch
import random
import pandas as pd
import glob
import soundfile as sf
import librosa
from scipy.optimize import fsolve


class SingleFileTrainPaired(torch.utils.data.IterableDataset):
    def __init__(
        self,
        file_clean=None,
        file_distorted=None,
        fs=44100,
        seg_len=131072,
        stereo=False,
        seed=42,
    ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        assert stereo == False, "Stereo not implemented yet"

        assert len(file_clean) == len(
            file_distorted
        ), "clean and distorted files have different lengths"

        for i, f in enumerate(file_clean):
            data_clean, samplerate = sf.read(file_clean[i])
            assert samplerate == fs, "wrong sampling rate"
            if len(data_clean.shape) > 1:
                data_clean = np.mean(data_clean, axis=1)

            if i == 0:
                self.data_clean = data_clean
            else:
                self.data_clean = np.concatenate((self.data_clean, data_clean), axis=0)

            data_distorted, samplerate = sf.read(file_distorted[i])
            assert samplerate == fs, "wrong sampling rate"
            if len(data_distorted.shape) > 1:
                data_distorted = np.mean(data_distorted, axis=1)

            if i == 0:
                self.data_distorted = data_distorted
            else:
                self.data_distorted = np.concatenate(
                    (self.data_distorted, data_distorted), axis=0
                )

        self.seg_len = int(seg_len)
        self.fs = fs

    def __iter__(self):
        while True:
            idx = np.random.randint(0, len(self.data_clean) - self.seg_len)
            segment_clean = self.data_clean[idx : idx + self.seg_len]
            segment_clean = segment_clean.astype("float32")
            segment_distorted = self.data_distorted[idx : idx + self.seg_len]
            segment_distorted = segment_distorted.astype("float32")
            yield segment_clean, segment_distorted




class SingleFileTrain(torch.utils.data.IterableDataset):
    def __init__(self, file=None, fs=44100, seg_len=131072, stereo=False, seed=42):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        assert stereo == False, "Stereo not implemented yet"
        print("file", file, len(file))
        for i, f in enumerate(file):
            data, samplerate = sf.read(file[i])
            assert samplerate == fs, "wrong sampling rate"
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            if i == 0:
                self.data = data
            else:
                self.data = np.concatenate((self.data, data), axis=0)

        self.seg_len = int(seg_len)
        self.fs = fs

    def __iter__(self):
        while True:
            idx = np.random.randint(0, len(self.data) - self.seg_len)
            segment = self.data[idx : idx + self.seg_len]
            segment = segment.astype("float32")
            yield segment




class EGDBtrain(torch.utils.data.IterableDataset):
    def __init__(
        self, path=None, path_csv=None, fs=44100, seg_len=131072, stereo=False, seed=42
    ):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        self.path = path
        assert stereo == False, "Stereo not implemented yet"
        filelist_df = pd.read_csv(path_csv)
        self.train_samples = filelist_df.values.tolist()

        self.seg_len = int(seg_len)
        self.fs = fs

    def __iter__(self):
        while True:
            num = random.randint(0, len(self.train_samples) - 1)
            file = os.path.join(self.path, self.train_samples[num][0])
            data, samplerate = sf.read(file)
            assert samplerate == self.fs, "wrong sampling rate"
            data_clean = data
            # Stereo to mono
            if len(data.shape) > 1:
                data_clean = np.mean(data_clean, axis=1)

            idx = np.random.randint(0, len(data_clean))
            if idx < len(data_clean) - self.seg_len:
                segment = data_clean[idx : idx + self.seg_len]
            else:
                data_clean = np.roll(data_clean, -self.seg_len)
                segment = data_clean[idx - self.seg_len : idx]
            segment = segment.astype("float32")
            yield segment







class SingleFileTest(torch.utils.data.Dataset):
    def __init__(
        self,
        fs=44100,
        segment_length=131072,
        file_clean="",
        file_distorted="",
        seed=42,
        mismatched=False,
        num_examples=-1,
        stereo=False,
        normalize=False,  # only applies to distorted
    ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.mismatched = mismatched
        assert len(file_clean) == len(
            file_distorted
        ), "clean and distorted files have different lengths"

        for i, f in enumerate(file_clean):

            data_clean, samplerate = sf.read(file_clean[i])

            assert samplerate == fs, "wrong sampling rate"

            if len(data_clean.shape) > 1:
                data_clean = np.mean(data_clean, axis=1)
            if i == 0:
                self.data_clean = data_clean
            else:
                # concatenate
                self.data_clean = np.concatenate((self.data_clean, data_clean), axis=0)

            data_distorted, samplerate = sf.read(file_distorted[i])

            if samplerate != fs:
                data_distorted = librosa.resample(
                    data_distorted, orig_sr=samplerate, target_sr=fs
                )
                samplerate = fs
            if len(data_distorted.shape) > 1:
                data_distorted = np.mean(data_distorted, axis=1)

            if i == 0:
                self.data_distorted = data_distorted
            else:
                self.data_distorted = np.concatenate(
                    (self.data_distorted, data_distorted), axis=0
                )

            if normalize != False:
                assert self.mismatched
                data_distorted = (
                    normalize * data_distorted / np.std(self.data_distorted)
                )

        if not self.mismatched:
            assert (
                self.data_clean.shape == self.data_distorted.shape
            ), "clean and distorted files have different lengths"


        self.seg_len = int(segment_length)
        self.fs = fs
        self.test_audio_clean = []
        self.test_audio_distorted = []
        self.filenames = []

        L = len(data_distorted)
        num_frames = int(np.floor(L / self.seg_len))
        if num_examples != -1:
            num_frames = max(num_examples, num_frames)

        for i in range(num_frames):
            idx = i * self.seg_len
            segment_clean = self.data_clean[idx : idx + self.seg_len]
            segment_distorted = self.data_distorted[idx : idx + self.seg_len]
            if segment_distorted.shape[0] < self.seg_len:
                # pad with zeros
                segment_distorted = np.pad(
                    segment_distorted,
                    (0, self.seg_len - segment_distorted.shape[0]),
                    "constant",
                )
            self.test_audio_clean.append(segment_clean)
            self.test_audio_distorted.append(segment_distorted)

        assert len(self.test_audio_clean) == len(
            self.test_audio_distorted
        ), "clean and distorted files have different lengths"


    def __getitem__(self, idx):
        return self.test_audio_clean[idx], self.test_audio_distorted[idx]

    def __len__(self):
        return len(self.test_audio_clean)




class FolderTestPaired(torch.utils.data.Dataset):
    def __init__(
        self,
        fs=44100,
        segment_length=131072,
        path_clean="",
        path_distorted="",
        seed=42,
        mismatched=False,
        num_examples=-1,
        stereo=False,
        normalize=False,  # only applies to distorted
        rms=0.05,
    ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        self.segment_length = segment_length
        print("Segment length", self.segment_length)
        print("Segment length", self.segment_length)

        self.mismatched = mismatched

        files_clean = glob.glob(os.path.join(path_clean, "*.wav"))
        print("files_clean", files_clean, path_clean)

        self.all_data_clean = []
        self.all_data_distorted = []

        for i, f in enumerate(files_clean):
            basename = os.path.basename(f)
            data_clean, samplerate = sf.read(files_clean[i])
            data_distorted, samplerate = sf.read(os.path.join(path_distorted, basename))

            if len(data_clean.shape) > 1:
                data_clean = np.mean(data_clean, axis=1)
            if len(data_distorted.shape) > 1:
                data_distorted = np.mean(data_distorted, axis=1)

            if samplerate != fs:
                print("resampling clean", samplerate, fs)
                data_clean = librosa.resample(
                    data_clean, orig_sr=samplerate, target_sr=fs
                )
                data_distorted = librosa.resample(
                    data_distorted, orig_sr=samplerate, target_sr=fs
                )

            if normalize != False:
                data_clean = rms * data_clean / np.std(data_clean)

            if data_clean.shape[-1] > data_distorted.shape[-1]:
                data_clean = data_clean[: data_distorted.shape[-1]]
            elif data_clean.shape[-1] < data_distorted.shape[-1]:
                data_distorted = data_distorted[: data_clean.shape[-1]]

            if segment_length != -1:
                L = len(data_clean)
                # crop or pad to get to the right length
                if L > self.segment_length:
                    # get random segment
                    idx = 0
                    data_clean = data_clean[idx : idx + self.segment_length]
                    data_distorted = data_distorted[idx : idx + self.segment_length]
                elif L <= self.segment_length:
                    # pad with zeros to get to the right length randomly
                    idx = 0
                    # copy segment to get to the right length
                    data_clean = np.pad(
                        data_clean, (idx, self.segment_length - L - idx), "wrap"
                    )
                    data_distorted = np.pad(
                        data_distorted, (idx, self.segment_length - L - idx), "wrap"
                    )

            self.all_data_clean.append(data_clean)
            self.all_data_distorted.append(data_distorted)

    def __getitem__(self, idx):
        return self.all_data_clean[idx], self.all_data_distorted[idx]

    def __len__(self):
        return len(self.all_data_clean)





class AudioFolderDatasetTest_numbered(torch.utils.data.Dataset):
    def __init__(
        self,
        fs=44100,
        segment_length=131072,
        path="",
        normalize=False,
        seed=42,
        num_examples=8,
        stereo=False,
    ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        filelist = glob.glob(os.path.join(path, "*.wav"))
        assert len(filelist) > 0, "error in dataloading: empty or nonexistent folder"

        filelist = sorted(
            filelist,
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]),
        )
        print("filelist", filelist)
        self.test_samples = filelist
        self.seg_len = int(segment_length)
        self.fs = fs
        self.test_samples = filelist

        assert (
            len(self.test_samples) >= num_examples
        ), "error in dataloading: not enough examples"

        self.normalize = normalize
        self.test_audio = []
        self.filenames = []
        self._fs = []

        for i in range(num_examples):
            file = self.test_samples[i]
            data, samplerate = sf.read(file)
            assert samplerate == self.fs, "wrong sampling rate"
            print("data", data.shape, self.test_samples[i])

            data_clean = data
            # Stereo to mono
            if len(data.shape) > 1:
                data_clean = np.mean(data_clean, axis=1)

            if normalize:
                data_clean = data_clean / np.std(data_clean)

            L = len(data_clean)

            # crop or pad to get to the right length
            if L > self.seg_len:
                idx = np.random.randint(0, L - self.seg_len)
                segment = data[idx : idx + self.seg_len]
            elif L < self.seg_len:
                raise ValueError("not implemented yet")
                # pad with zeros to get to the right length randomly
                idx = np.random.randint(0, self.segment_length - L)
                segment = np.pad(data, (idx, self.segment_length - L - idx), "wrap")
            elif L == self.seg_len:
                segment = data

            self.test_audio.append(segment)
            self.filenames.append(file)

    def __getitem__(self, idx):
        return self.test_audio[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)

