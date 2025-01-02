---
date: '2025-01-02T13:33:03+01:00'
draft: true
title: 'Guitar evaluation'
summary: 'Hard/Soft clipping for speech signals versus baselines'
weight: 2
---


## Hard Clipping

{{< shared-audio-player >}}

| Method         |                                                  1dB                                                  |                    3dB                     |     7dB     |
|:---------------|:-----------------------------------------------------------------------------------------------------:|:------------------------------------------:|:-----------:|
| **Clean**      | {{< playbutton src="/audio/clean_0.wav">}}   {{< playbutton src="/audio/clean_0.wav">}}   |
| **Clipped**    |                               {{< playbutton src="/audio/inp_0.wav">}}                                |                Here's this                 | Here's this |
| **Informed**   |                                                  TBD                                                  |                  And more                  |  And more   |
| **Supervised** |                                {{< playbutton src="/audio/x_0.wav">}}                                 |                  And more                  |  And more   |
| **SumTanh**    |                                                 Text                                                  |                  And more                  |  And more   |
| **MLP**        |                                                 Text                                                  |                  And more                  |  And more   |
| **CCR**        |                                                 Text                                                  |                  And more                  |  And more   |
 

## Soft Clipping


