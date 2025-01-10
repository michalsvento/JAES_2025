import os
import numpy as np
import torch
import random
import glob
import soundfile as sf
from torchvision.transforms import Compose


class VCTKTrain(torch.utils.data.IterableDataset):
    def __init__(
        self,
        fs=16000,
        segment_length=65536,
        path="",  # path to the dataset
        speakers_discard=[],  # list of speakers to discard
        speakers_test=[],  # list of speakers to use for testing, discarded here
        normalize=False,  # to normalize or not. I don't normalize by default
        seed=0,
        extension=".wav",
        rms=0.05,
    ):

        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.train_samples = []
        # iterate over speakers directories
        speakers = os.listdir(path)
        for s in speakers:
            if s in speakers_discard:
                continue
            elif s in speakers_test:
                continue
            else:
                self.train_samples.extend(
                    glob.glob(os.path.join(path, s, "*" + extension))
                )

        assert (
            len(self.train_samples) > 0
        ), "error in dataloading: empty or nonexistent folder"
        self.segment_length = int(segment_length)
        self.fs = fs
        self.normalize = normalize
        if self.normalize:
            self.rms = rms

    def __iter__(self):

        while True:
            num = random.randint(0, len(self.train_samples) - 1)
            file = self.train_samples[num]
            data, samplerate = sf.read(file)
            assert samplerate == self.fs, "wrong sampling rate"
            segment = data
            # Stereo to mono
            if len(data.shape) > 1:
                segment = np.mean(segment, axis=1)

            L = len(segment)
            # crop or pad to get to the right length
            if L > self.segment_length:
                # get random segment
                idx = np.random.randint(0, L - self.segment_length)
                segment = segment[idx : idx + self.segment_length]
            elif L <= self.segment_length:
                # pad with zeros to get to the right length randomly
                idx = np.random.randint(0, self.segment_length - L)
                # copy segment to get to the right length
                segment = np.pad(segment, (idx, self.segment_length - L - idx), "wrap")

            if self.normalize:
                segment = self.rms * segment / np.sqrt(np.mean(segment**2))

            yield segment.astype(np.float32)


class VCTKTest(torch.utils.data.Dataset):
    def __init__(
        self,
        fs=16000,
        segment_length=65536,
        path="",  # path to the dataset
        speakers_discard=[],  # list of speakers to discard
        speakers_test=[],  # list of speakers to use for testing, discarded here
        normalize=False,  # to normalize or not. I don't normalize by default
        seed=0,
        num_examples=8,
        shuffle=True,
        extension=".wav",
        rms=0.05,
    ):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.test_samples = []
        # iterate over speakers directories
        speakers = os.listdir(path)
        for s in speakers:
            if s in speakers_discard:
                continue
            elif s in speakers_test:
                self.test_samples.extend(
                    glob.glob(os.path.join(path, s, "*" + extension))
                )
            else:
                continue

        self.test_samples = sorted(self.test_samples)
        assert (
            len(self.test_samples) >= num_examples
        ), "error in dataloading: not enough examples"

        if num_examples > 0:
            if shuffle:
                self.test_samples = random.sample(self.test_samples, num_examples)
            else:
                self.test_samples = self.test_samples[:num_examples]

        self.segment_length = int(segment_length)
        self.fs = fs
        self.normalize = normalize
        if self.normalize:
            self.rms = rms

        self.test_audio = []
        self.filenames = []
        self._fs = []
        for file in self.test_samples:
            self.filenames.append(os.path.basename(file))
            data, samplerate = sf.read(file)
            assert samplerate == self.fs, "wrong sampling rate"
            assert len(data.shape) == 1, "wrong number of channels"
            L = len(data)

            if self.segment_length > 0:
                # crop or pad to get to the right length
                if L > self.segment_length:
                    # get random segment
                    idx = np.random.randint(0, L - self.segment_length)
                    segment = data[idx : idx + self.segment_length]
                elif L <= self.segment_length:
                    # pad with zeros to get to the right length randomly
                    idx = np.random.randint(0, self.segment_length - L)
                    # copy segment to get to the right length
                    segment = np.pad(data, (idx, self.segment_length - L - idx), "wrap")
            else:
                segment = data

            segment = rms * segment / np.sqrt(np.mean(segment**2))

            self.test_audio.append(segment)  # use only 50s

    def __getitem__(self, idx):
        return self.test_audio[idx], self.filenames[idx]

    def __len__(self):
        return len(self.test_samples)
