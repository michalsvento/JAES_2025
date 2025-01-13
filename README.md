# Estimation and Restoration of Unknown Nonlinear Distortion using Diffusion

This is the official supplementary code and webpage for same name paper.

#### [webpage 🎶](https://michalsvento.github.io/NLDistortionDiff/) |  [paper 📰](http://arxiv.org/abs/2501.05959)


## Abstract


The restoration of nonlinearly distorted audio signals,
alongside the identification of the applied memoryless nonlinear operation, is studied.
The paper focuses on the difficult but practically important case in which both the nonlinearity and the original input signal are unknown.
The proposed method uses a generative diffusion model trained unconditionally on guitar or speech signals to jointly model and invert the nonlinear system at inference time.
Both the memoryless nonlinear function model and the restored audio signal are obtained as output.
Successful example case studies are presented including inversion of hard and soft clipping,
digital quantization, half-wave rectification, and wavefolding nonlinearities.
Our results suggest that, out of the nonlinear functions tested here,
the cubic Catmull-Rom spline is best suited to approximating these nonlinearities.
In the case of guitar recordings,
comparisons with informed and supervised methods show that the proposed blind method is at least as good as they are in terms of objective metrics.
Experiments on distorted speech show that the proposed blind method outperforms general-purpose speech enhancement techniques and restores the original voice quality.
The proposed method can be applied to audio effects modeling, restoration of music and speech recordings,
and characterization of analog recording media.


## Code


### Installation

We tested the code on Python 3.11.10. 
We recommend creating empty environment.
To install the required packages, run:

```
pip install -r requirements.txt
```

### Checkpoints and test set

We uploaded the checkpoints and test set as a release.
You can download them from the [release page](https://github.com/michalsvento/NLDistortionDiff/releases/tag/checkpoints).


### Inference

For inference, you need to modify the `main_test_guitar.sh`/`main_test_speech.sh`.

- Set your environment
- Uncomment config file for specific distortion
- Update the path to the test set in the config file
- Set the path to the checkpoint
- Set up wandb

Then you can run the inference script:

```
bash main_test_guitar.sh
```

### Training

For training, you need to modify the `dset` config file and update the paths to your files.
    
Then you can run the training script:
    
```
bash train_guitar.sh
```


## Citing

If you find this code useful in your research, please consider citing:

```
@article{svento2025NLDistortionDiff,
      title={Estimation and Restoration of Unknown Nonlinear Distortion using Diffusion}, 
      author={Michal Švento and Eloi Moliner and Lauri Juvela and Alec Wright and Vesa Välimäki},
      year={2025},
      journal={arxiv 2501.05959},
}
```

