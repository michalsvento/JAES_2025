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


class SingleFileTrainGenerateOnFly(torch.utils.data.IterableDataset):
    def __init__(
        self,
        file_clean=None,
        fs=44100,
        seg_len=131072,
        stereo=False,
        seed=42,
        distortion="hardclip",
    ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        # self.path=path
        assert stereo == False, "Stereo not implemented yet"

        for i, f in enumerate(file_clean):
            data_clean, samplerate = sf.read(file_clean[i])
            assert samplerate == fs, "wrong sampling rate"
            if len(data_clean.shape) > 1:
                data_clean = np.mean(data_clean, axis=1)

            if i == 0:
                self.data_clean = data_clean
            else:
                self.data_clean = np.concatenate((self.data_clean, data_clean), axis=0)

        self.seg_len = int(seg_len)
        self.fs = fs
        self.distortion = distortion

        def adapt_params_to_SDR(ref, SDRtarget, distortion=None, k=None):
            x = ref
            if distortion == "hardclip":
                hardclip = lambda x, limit: np.clip(x, -limit, limit)
                nld = hardclip
            if distortion == "general":

                def softclip(x, limit, k=0.7):
                    r = k * limit
                    y = np.where(
                        np.abs(x) <= r,
                        x,
                        np.sign(x)
                        * ((limit - r) * np.tanh((np.abs(x) - r) / (limit - r)) + r),
                    )
                    return y

                nld = lambda x, limit: softclip(x, limit, k)

            def find_clip_value(thresh):
                xclipped = nld(x, thresh)
                sdr = 20 * np.log10(
                    np.linalg.norm(x) / (np.linalg.norm(x - xclipped) + 1e-7)
                )
                return np.abs(sdr - SDRtarget)

            clip_value = fsolve(find_clip_value, 0.01)
            y = nld(x, clip_value)
            return y

        self.find_y = adapt_params_to_SDR

    def __iter__(self):
        while True:
            idx = np.random.randint(0, len(self.data_clean) - self.seg_len)
            segment_clean = self.data_clean[idx : idx + self.seg_len]
            alpha = 0.002 * torch.randn(1) + 0.06
            alpha = alpha.numpy()
            segment_clean = (
                segment_clean.astype("float32") * alpha / segment_clean.std()
            )

            n = np.random.uniform(0, 1)
            if n > 0.5:
                segment_clean = -segment_clean
            target_SDR = np.random.uniform(0.5, 10)
            k = np.random.uniform(0, 1)
            segment_distorted = self.find_y(
                segment_clean, target_SDR, self.distortion, k
            )
            segment_distorted = segment_distorted.astype("float32")
            yield segment_distorted, segment_clean


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


class classical_music_dataset_train(torch.utils.data.IterableDataset):
    def __init__(
        self,
        datasets=None,
        priorities=None,
        path=None,
        path_csv=None,
        fs=44100,
        seg_len=131072,
        stereo=False,
        seed=42,
    ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.path = path
        assert stereo == False, "Stereo not implemented yet"

        self.datasets = datasets
        self.priorities = priorities
        self.priorities /= np.sum(self.priorities)
        self.priorities_cumsum = np.cumsum(self.priorities)
        assert self.priorities is not None

        self.path = path
        self.path_csv = path_csv

        assert len(datasets) == len(
            priorities
        ), "datasets and priorities have different lengths"
        assert len(datasets) > 0, "no datasets provided"
        assert len(datasets) == len(path), "datasets and paths have different lengths"
        assert len(datasets) == len(
            path_csv
        ), "datasets and paths have different lengths"

        self.filelists = []
        for i in range(len(datasets)):
            filelist_i = pd.read_csv(self.path_csv[i], header=None)
            assert len(filelist_i) > 0, "empty filelist"
            print(len(filelist_i), datasets[i])
            self.filelists.append(filelist_i)

        self.seg_len = int(seg_len)
        self.fs = fs

    def __iter__(self):
        while True:
            try:
                n = random.random()
                idx_dset = np.searchsorted(self.priorities_cumsum, n)
                idx_dset = int(idx_dset)

                filelist = self.filelists[idx_dset]
                idx_file = random.randint(0, len(filelist) - 1)
                idx_file = int(idx_file)

                file = os.path.join(
                    self.path[idx_dset], filelist.loc[idx_file].values[0]
                )

                data, samplerate = sf.read(file)
                assert samplerate == self.fs, "wrong sampling rate"
                data_clean = data
                # Stereo to mono
                if len(data.shape) > 1:
                    data_clean = np.mean(data_clean, axis=1)

                done = False
                counter = 0
                while not (done):
                    idx = np.random.randint(0, len(data_clean))
                    if idx < len(data_clean) - self.seg_len:
                        segment = data_clean[idx : idx + self.seg_len]
                    else:
                        data_clean = np.roll(data_clean, -self.seg_len)
                        segment = data_clean[idx - self.seg_len : idx]

                    if segment.std() > 1e-3:
                        done = True
                    else:
                        "rms is too low"
                        counter += 1

                    if counter > 5:
                        print(" skipping file")
                        break

                if done:
                    segment = segment.astype("float32")
                    yield segment
            except:
                pass


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


class AudioFolderDataset(torch.utils.data.IterableDataset):
    def __init__(self, dset_args, fs=44100, seg_len=131072, overfit=False, seed=42):
        self.overfit = overfit
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        path = dset_args.path

        filelist = glob.glob(os.path.join(path, "*.wav"))
        assert len(filelist) > 0, "error in dataloading: empty or nonexistent folder"
        self.train_samples = filelist
        self.seg_len = int(seg_len)
        self.fs = fs
        if self.overfit:
            file = self.train_samples[0]
            data, samplerate = sf.read(file)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            self.overfit_sample = data[
                10 * samplerate : 60 * samplerate
            ]  # use only 50s

    def __iter__(self):
        if self.overfit:
            data_clean = self.overfit_sample
        while True:
            if not self.overfit:
                num = random.randint(0, len(self.train_samples) - 1)
                file = self.train_samples[num]
                data, samplerate = sf.read(file)
                assert samplerate == self.fs, "wrong sampling rate"
                data_clean = data
                # Stereo to mono
                if len(data.shape) > 1:
                    data_clean = np.mean(data_clean, axis=1)

            num_frames = np.floor(len(data_clean) / self.seg_len)

            for i in range(8):
                if not self.overfit:
                    idx = np.random.randint(0, len(data_clean) - self.seg_len)
                else:
                    idx = 0
                segment = data_clean[idx : idx + self.seg_len]
                segment = segment.astype("float32")
                yield segment


class SetTest(torch.utils.data.Dataset):
    def __init__(
        self,
        fs=44100,
        segment_length=131072,
        path="",
        path_csv=None,
        seed=42,
        num_examples=8,
        stereo=False,
        do_random=False,
        idx_start_seconds=0,
    ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        self.do_random = do_random
        self.idx_start = idx_start_seconds * fs
        self.path = path
        filelist_df = pd.read_csv(path_csv, header=None)
        # convert the dataframe to list
        self.test_samples = filelist_df.values.tolist()
        self.seg_len = int(segment_length)
        self.fs = fs
        assert (
            len(self.test_samples) >= num_examples
        ), "error in dataloading: not enough examples"

        # take num_examples random examples
        if num_examples > 0:
            self.test_samples = random.sample(self.test_samples, num_examples)

        self.test_audio = []
        self.filenames = []
        self._fs = []

        for i in range(len(self.test_samples)):
            # for file in self.train_samples:
            file = os.path.join(self.path, self.test_samples[i][0])
            data, samplerate = sf.read(file)
            assert samplerate == self.fs, "wrong sampling rate"

            data_clean = data
            # Stereo to mono
            if len(data.shape) > 1:
                data_clean = np.mean(data_clean, axis=1)

            L = len(data_clean)
            segment = data_clean
            self.test_audio.append(segment)
            self.filenames.append(file)

    def __getitem__(self, idx):
        return self.test_audio[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)


class MultipleFilesTest(torch.utils.data.Dataset):
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

        print("file clean", file_clean)
        for i, f in enumerate(file_clean):
            print(file_clean[i])
            data_clean, samplerate = sf.read(file_clean[i])
            print(data_clean.shape)
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

        print("self.data_clean", self.data_clean.shape)
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

        print("len", len(self.test_audio_clean))
        print("len", len(self.test_audio_distorted))

    def __getitem__(self, idx):
        return self.test_audio_clean[idx], self.test_audio_distorted[idx]

    def __len__(self):
        return len(self.test_audio_clean)


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

        print("file clean", file_clean)
        for i, f in enumerate(file_clean):
            print(file_clean[i])
            data_clean, samplerate = sf.read(file_clean[i])
            print(data_clean.shape)
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

        print("self.data_clean", self.data_clean.shape)
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

        print("len", len(self.test_audio_clean))
        print("len", len(self.test_audio_distorted))

    def __getitem__(self, idx):
        return self.test_audio_clean[idx], self.test_audio_distorted[idx]

    def __len__(self):
        return len(self.test_audio_clean)


class SingleFileTest_Effected(torch.utils.data.Dataset):
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


class SingleFileTestDistorted(torch.utils.data.Dataset):
    def __init__(
        self,
        fs=44100,
        seg_len=131072,
        file_clean="",
        seed=42,
        mismatched=False,
        num_examples=-1,
        stereo=False,
        distortion="hardclip",
    ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        self.mismatched = mismatched

        for i, f in enumerate(file_clean):
            data_clean, samplerate = sf.read(file_clean[i])

            self.data_clean = data_clean
            if len(data_clean.shape) > 1:
                self.data_clean = np.mean(self.data_clean, axis=1)

            if samplerate != fs:
                print("resampling clean", samplerate, fs)
                self.data_clean = librosa.resample(
                    self.data_clean, orig_sr=samplerate, target_sr=fs
                )

        segment_length = seg_len
        self.seg_len = int(segment_length)
        self.fs = fs

        self.test_audio_clean = []
        self.test_audio_distorted = []
        self.filenames = []
        L_clean = len(self.data_clean)

        num_frames = int(np.floor(L_clean / self.seg_len))
        if num_examples != -1:
            num_frames = min(num_examples, num_frames)

        # print("loading ")
        for i in range(num_frames):
            idx = i * self.seg_len
            segment_clean = self.data_clean[idx : idx + self.seg_len]

            SDR = 2

            segment_distorted = self.adapt_params_to_SDR(segment_clean, SDR, "hardclip")

            if segment_clean.shape[0] < self.seg_len:
                segment_clean = np.pad(
                    segment_clean,
                    (0, self.seg_len - segment_clean.shape[0]),
                    "constant",
                )

            self.test_audio_clean.append(segment_clean)
            self.test_audio_distorted.append(segment_distorted)

        assert len(self.test_audio_clean) == len(
            self.test_audio_distorted
        ), "clean and distorted files have different lengths"

    def adapt_params_to_SDR(self, ref, SDRtarget, distortion=None, k=None):
        x = ref

        if distortion == "hardclip":
            hardclip = lambda x, limit: np.clip(x, -limit, limit)
            nld = hardclip
        if distortion == "general":
            assert k is not None

            def softclip(x, limit, k=0.7):
                r = k * limit
                y = np.where(
                    np.abs(x) <= r,
                    x,
                    np.sign(x)
                    * ((limit - r) * np.tanh((np.abs(x) - r) / (limit - r)) + r),
                )
                return y

            nld = lambda x, limit: softclip(x, limit, k)

        def find_clip_value(thresh):
            xclipped = nld(x, thresh)
            sdr = 20 * np.log10(
                np.linalg.norm(x) / (np.linalg.norm(x - xclipped) + 1e-7)
            )
            return np.abs(sdr - SDRtarget)

        clip_value = fsolve(find_clip_value, 0.01)
        y = nld(x, clip_value)
        return y

    def __getitem__(self, idx):
        return self.test_audio_clean[idx], self.test_audio_distorted[idx]

    def __len__(self):
        return len(self.test_audio_clean)


class EGDBtest(torch.utils.data.Dataset):
    def __init__(
        self,
        fs=44100,
        segment_length=131072,
        path="",
        path_csv=None,
        seed=42,
        num_examples=8,
        stereo=False,
        do_random=False,
        idx_start_seconds=0,
    ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.do_random = do_random

        self.idx_start = idx_start_seconds * fs

        self.path = path
        filelist_df = pd.read_csv(path_csv)
        # convert the dataframe to list
        self.test_samples = filelist_df.values.tolist()
        self.seg_len = int(segment_length)
        self.fs = fs

        assert (
            len(self.test_samples) >= num_examples
        ), "error in dataloading: not enough examples"

        # take num_examples random examples
        print("num_test_samples", len(self.test_samples))
        self.test_samples = random.sample(self.test_samples, num_examples)
        print("num_test_samples", len(self.test_samples))

        self.test_audio = []
        self.filenames = []
        self._fs = []

        for i in range(num_examples):
            # for file in self.train_samples:
            file = os.path.join(self.path, self.test_samples[i][0])
            data, samplerate = sf.read(file)
            assert samplerate == self.fs, "wrong sampling rate"

            data_clean = data
            # Stereo to mono
            if len(data.shape) > 1:
                data_clean = np.mean(data_clean, axis=1)

            L = len(data_clean)

            # crop or pad to get to the right length
            if L > self.seg_len:
                # get random segment
                if self.do_random:
                    idx = np.random.randint(0, L - self.seg_len)
                else:
                    idx = self.idx_start
                segment = data[idx : idx + self.seg_len]
            elif L <= self.segment_length:
                raise ValueError("not implemented yet")
                # pad with zeros to get to the right length randomly
                idx = np.random.randint(0, self.segment_length - L)
                segment = np.pad(data, (idx, self.segment_length - L - idx), "wrap")

            self.test_audio.append(segment)
            self.filenames.append(file)

    def __getitem__(self, idx):
        return self.test_audio[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)


class AudioFolderTest(torch.utils.data.Dataset):
    def __init__(
        self,
        fs=44100,
        segment_length=131072,
        path="",
        normalize=False,
        seed=42,
        num_examples=8,
        stereo=False,
        length_power_of_two=False,
    ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        filelist = glob.glob(os.path.join(path, "*.wav"))
        assert len(filelist) > 0, "error in dataloading: empty or nonexistent folder"

        self.test_samples = filelist
        self.seg_len = int(segment_length)
        self.fs = fs

        self.test_samples = filelist
        # print("test_samples", self.test_samples)

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

            if self.seg_len != -1:
                if length_power_of_two:
                    assert data_clean.shape[-1] & (data_clean.shape[-1] - 1) == 0
                    # find the next power of 2
                    next_power_of_2 = 1 << data_clean.shape[-1].bit_length()
                    # pad the input to the next power of 2
                    data_clean = np.pad(
                        data_clean, (0, next_power_of_2 - data_clean.shape[-1])
                    )

                # crop or pad to get to the right length
                if L > self.seg_len:
                    # get random segment
                    idx = 0
                    segment = data_clean[idx : idx + self.seg_len]
                elif L < self.seg_len:
                    idx = 0
                    # copy segment to get to the right length
                    segment = np.pad(data_clean, (idx, self.seg_len - L - idx), "wrap")
                elif L == self.seg_len:
                    segment = data_clean
            else:
                if length_power_of_two:
                    if not (data_clean.shape[-1] & (data_clean.shape[-1] - 1) == 0):
                        # find the next power of 2
                        next_power_of_2 = 1 << data_clean.shape[-1].bit_length()
                        # pad the input to the next power of 2
                        data = np.pad(
                            data_clean, (0, next_power_of_2 - data_clean.shape[-1])
                        )
                segment = data

            self.test_audio.append(segment)
            self.filenames.append(file)

    def __getitem__(self, idx):
        return self.test_audio[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)


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


class NoisePlusDataRFlow(torch.utils.data.IterableDataset):
    def __init__(self, path=None, fs=44100, seg_len=131072, seed=42, stereo=False):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)
        path = path
        print(path)
        dpath = glob.glob(os.path.join(path, "unconditional_*.wav"))
        print(dpath)

        datalist = sorted(glob.glob(os.path.join(path, "unconditional_*.wav")))
        noiselist = sorted(glob.glob(os.path.join(path, "noise_*.wav")))

        assert len(datalist) > 0, "error in dataloading: empty or nonexistent folder"
        assert len(noiselist) > 0, "error in dataloading: empty or nonexistent folder"
        assert len(datalist) == len(noiselist), "Data should have the noise pair"

        print("datalist", datalist)
        print("noiselist", noiselist)

        self.data_samples = datalist
        self.noise_samples = noiselist
        self.seg_len = int(seg_len)
        self.fs = fs

        print("len data_samples", len(self.data_samples))
        print("len noise_samples", len(self.noise_samples))

    def __iter__(self):
        while True:
            num = random.randint(0, len(self.data_samples) - 1)
            data_clean, samplerate = sf.read(self.data_samples[num])
            data_noise, noise_sr = sf.read(self.noise_samples[num])
            assert samplerate == self.fs, "wrong sampling rate on datasite"
            assert noise_sr == self.fs, "wrong sampling rate on noise site"

            idx = np.random.randint(0, len(data_clean) - self.seg_len)
            segment = data_clean[idx : idx + self.seg_len]
            noise_segment = data_noise[idx : idx + self.seg_len]
            segment = segment.astype("float32")
            noise_segment = noise_segment.astype("float32")
            yield segment, noise_segment
