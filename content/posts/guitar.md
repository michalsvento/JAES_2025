---
date: '2025-01-02T13:33:03+01:00'
draft: False
title: 'Guitar evaluation'
summary: 'Hard/Soft clipping for speech signals versus baselines'
weight: 3
---


## Hard Clipping

| Audiofile |                     0                      | 2 | 5 |
|:----------|:------------------------------------------:|:-:|:-:|
| Select | x | x | x |
| **Clean** | {{< playbutton src="/audio/clean_0.wav">}} | {{< playbutton src="/audio/clean_0.wav">}} |{{< playbutton src="/audio/clean_0.wav">}} |
    

{{< shared-audio-player id="hard-clipping-player">}}

| Method         |                                                  1dB                                                  |                    3dB                     |  7dB  |
|:---------------|:-----------------------------------------------------------------------------------------------------:|:------------------------------------------:|:-----:|
| **Clipped**    |                               {{< playbutton src="/audio/inp_0.wav">}}                                |                TBD                |  TBD  |
| **Informed**   |                                                  TBD                                                  |                TBD                |  TBD  |
| **Supervised** |                                {{< playbutton src="/audio/x_0.wav">}}                                 |                TBD                |  TBD  |
| **SumTanh**    |                                                 TBD                                                  |                 TBD               |  TBD  |
| **MLP**        |                                                 TBD                                                  |                 TBD               |  TBD  |
| **CCR**        |                                                 TBD                                                  |                 TBD               |  TBD  |
 

## Soft Clipping

{{< shared-audio-player id="hard-clipping-player">}}
