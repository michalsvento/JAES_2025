---
date: '2025-01-02T13:33:03+01:00'
draft: False
title: 'Abstract'
#summary: 'Hard/Soft clipping for speech signals versus baselines'
hideSummary: true
weight: 1
---

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