---
date: '2025-01-02T14:09:49+01:00'
draft: False
title: 'Speech Evaluation'
# description: 'This is speech eval'
# hideSummary: true
summary: 'Hard/Soft clipping for speech signals versus baselines'
weight: 4
hidemeta: true
---

## Hard clipping

### 1dB

{{< shared-audio-player id="1" >}}

S1, S2: two speakers

|            |                                S1  1                                 |                                 S1 2                                 |                                 S2 1                                 |                                 S2 2                                 |
|:----------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|
|   Clean    |  {{< playbutton src="speech/hardclip/1db/original/p232_013.wav" >}}  |  {{< playbutton src="speech/hardclip/1db/original/p232_025.wav" >}}  |  {{< playbutton src="speech/hardclip/1db/original/p257_015.wav" >}}  |  {{< playbutton src="speech/hardclip/1db/original/p257_024.wav" >}}  |
|  Clipped   |  {{< playbutton src="speech/hardclip/1db/clipped/p232_013.wav" >}}  | {{< playbutton src="speech/hardclip/1db/clipped/p232_025.wav" >}}  | {{< playbutton src="speech/hardclip/1db/clipped/p257_015.wav" >}}  | {{< playbutton src="speech/hardclip/1db/clipped/p257_024.wav" >}}  |
|  A-SPADE   |   {{< playbutton src="speech/hardclip/1db/aspade/p232_013.wav" >}}   |   {{< playbutton src="speech/hardclip/1db/aspade/p232_025.wav" >}}   |   {{< playbutton src="speech/hardclip/1db/aspade/p257_015.wav" >}}   |   {{< playbutton src="speech/hardclip/1db/aspade/p257_024.wav" >}}   |
|  Resemble  |  {{< playbutton src="speech/hardclip/1db/resemble/p232_013.wav" >}}  |  {{< playbutton src="speech/hardclip/1db/resemble/p232_025.wav" >}}  |  {{< playbutton src="speech/hardclip/1db/resemble/p257_015.wav" >}}  |  {{< playbutton src="speech/hardclip/1db/resemble/p257_024.wav" >}}  |
| Voicefixer | {{< playbutton src="speech/hardclip/1db/voicefixer/p232_013.wav" >}} | {{< playbutton src="speech/hardclip/1db/voicefixer/p232_025.wav" >}} | {{< playbutton src="speech/hardclip/1db/voicefixer/p257_015.wav" >}} | {{< playbutton src="speech/hardclip/1db/voicefixer/p257_024.wav" >}} |
|    DDD     |    {{< playbutton src="speech/hardclip/1db/DDD/p232_013.wav" >}}     |    {{< playbutton src="speech/hardclip/1db/DDD/p232_025.wav" >}}     |    {{< playbutton src="speech/hardclip/1db/DDD/p257_015.wav" >}}     |    {{< playbutton src="speech/hardclip/1db/DDD/p257_024.wav" >}}     |
|  informed  |  {{< playbutton src="speech/hardclip/1db/informed/p232_013.wav" >}}  |  {{< playbutton src="speech/hardclip/1db/informed/p232_025.wav" >}}  |  {{< playbutton src="speech/hardclip/1db/informed/p257_015.wav" >}}  |  {{< playbutton src="speech/hardclip/1db/informed/p257_024.wav" >}}  |
|    MLP     |    {{< playbutton src="speech/hardclip/1db/mlp/p232_013.wav" >}}     |    {{< playbutton src="speech/hardclip/1db/mlp/p232_025.wav" >}}     |    {{< playbutton src="speech/hardclip/1db/mlp/p257_015.wav" >}}     |    {{< playbutton src="speech/hardclip/1db/mlp/p257_024.wav" >}}     |
|    CCR     |    {{< playbutton src="speech/hardclip/1db/ccr/p232_013.wav" >}}     |    {{< playbutton src="speech/hardclip/1db/ccr/p232_025.wav" >}}     |    {{< playbutton src="speech/hardclip/1db/ccr/p257_015.wav" >}}     |    {{< playbutton src="speech/hardclip/1db/ccr/p257_024.wav" >}}     |


### 3dB


|            |                                S1  1                                 |                                 S1 2                                 |                                 S2 1                                 |                                 S2 2                                 |
|:----------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|
|   Clean    |  {{< playbutton src="speech/hardclip/3db/original/p232_013.wav" >}}  |  {{< playbutton src="speech/hardclip/3db/original/p232_025.wav" >}}  |  {{< playbutton src="speech/hardclip/3db/original/p257_015.wav" >}}  |  {{< playbutton src="speech/hardclip/3db/original/p257_024.wav" >}}  |
|  Clipped   | {{< playbutton src="speech/hardclip/3db/clipped/p232_013.wav" >}}  | {{< playbutton src="speech/hardclip/3db/clipped/p232_025.wav" >}}  | {{< playbutton src="speech/hardclip/3db/clipped/p257_015.wav" >}}  | {{< playbutton src="speech/hardclip/3db/clipped/p257_024.wav" >}}  |
|  A-SPADE   |   {{< playbutton src="speech/hardclip/3db/aspade/p232_013.wav" >}}   |   {{< playbutton src="speech/hardclip/3db/aspade/p232_025.wav" >}}   |   {{< playbutton src="speech/hardclip/3db/aspade/p257_015.wav" >}}   |   {{< playbutton src="speech/hardclip/3db/aspade/p257_024.wav" >}}   |
|  Resemble  |  {{< playbutton src="speech/hardclip/3db/resemble/p232_013.wav" >}}  |  {{< playbutton src="speech/hardclip/3db/resemble/p232_025.wav" >}}  |  {{< playbutton src="speech/hardclip/3db/resemble/p257_015.wav" >}}  |  {{< playbutton src="speech/hardclip/3db/resemble/p257_024.wav" >}}  |
| Voicefixer | {{< playbutton src="speech/hardclip/3db/voicefixer/p232_013.wav" >}} | {{< playbutton src="speech/hardclip/3db/voicefixer/p232_025.wav" >}} | {{< playbutton src="speech/hardclip/3db/voicefixer/p257_015.wav" >}} | {{< playbutton src="speech/hardclip/3db/voicefixer/p257_024.wav" >}} |
|    DDD     |    {{< playbutton src="speech/hardclip/3db/DDD/p232_013.wav" >}}     |    {{< playbutton src="speech/hardclip/3db/DDD/p232_025.wav" >}}     |    {{< playbutton src="speech/hardclip/3db/DDD/p257_015.wav" >}}     |    {{< playbutton src="speech/hardclip/3db/DDD/p257_024.wav" >}}     |
|  Informed  |  {{< playbutton src="speech/hardclip/3db/informed/p232_013.wav" >}}  |  {{< playbutton src="speech/hardclip/3db/informed/p232_025.wav" >}}  |  {{< playbutton src="speech/hardclip/3db/informed/p257_015.wav" >}}  |  {{< playbutton src="speech/hardclip/3db/informed/p257_024.wav" >}}  |
|    MLP     |    {{< playbutton src="speech/hardclip/3db/mlp/p232_013.wav" >}}     |    {{< playbutton src="speech/hardclip/3db/mlp/p232_025.wav" >}}     |    {{< playbutton src="speech/hardclip/3db/mlp/p257_015.wav" >}}     |    {{< playbutton src="speech/hardclip/3db/mlp/p257_024.wav" >}}     |
|    CCR     |    {{< playbutton src="speech/hardclip/3db/ccr/p232_013.wav" >}}     |    {{< playbutton src="speech/hardclip/3db/ccr/p232_025.wav" >}}     |    {{< playbutton src="speech/hardclip/3db/ccr/p257_015.wav" >}}     |    {{< playbutton src="speech/hardclip/3db/ccr/p257_024.wav" >}}     |

### 7dB

|            |                                S1  1                                 |                                 S1 2                                 |                                 S2 1                                 |                                 S2 2                                 |
|:----------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|
|   Clean    |  {{< playbutton src="speech/hardclip/7db/original/p232_013.wav" >}}  |  {{< playbutton src="speech/hardclip/7db/original/p232_025.wav" >}}  |  {{< playbutton src="speech/hardclip/7db/original/p257_015.wav" >}}  |  {{< playbutton src="speech/hardclip/7db/original/p257_024.wav" >}}  |
|  Clipped   |  {{< playbutton src="speech/hardclip/7db/clipped/p232_013.wav" >}}  | {{< playbutton src="speech/hardclip/7db/clipped/p232_025.wav" >}}  | {{< playbutton src="speech/hardclip/7db/clipped/p257_015.wav" >}}  | {{< playbutton src="speech/hardclip/7db/clipped/p257_024.wav" >}}  |
|  A-SPADE   |   {{< playbutton src="speech/hardclip/7db/aspade/p232_013.wav" >}}   |   {{< playbutton src="speech/hardclip/7db/aspade/p232_025.wav" >}}   |   {{< playbutton src="speech/hardclip/7db/aspade/p257_015.wav" >}}   |   {{< playbutton src="speech/hardclip/7db/aspade/p257_024.wav" >}}   |
|  Resemble  |  {{< playbutton src="speech/hardclip/7db/resemble/p232_013.wav" >}}  |  {{< playbutton src="speech/hardclip/7db/resemble/p232_025.wav" >}}  |  {{< playbutton src="speech/hardclip/7db/resemble/p257_015.wav" >}}  |  {{< playbutton src="speech/hardclip/7db/resemble/p257_024.wav" >}}  |
| Voicefixer | {{< playbutton src="speech/hardclip/7db/voicefixer/p232_013.wav" >}} | {{< playbutton src="speech/hardclip/7db/voicefixer/p232_025.wav" >}} | {{< playbutton src="speech/hardclip/7db/voicefixer/p257_015.wav" >}} | {{< playbutton src="speech/hardclip/7db/voicefixer/p257_024.wav" >}} |
|    DDD     |    {{< playbutton src="speech/hardclip/7db/DDD/p232_013.wav" >}}     |    {{< playbutton src="speech/hardclip/7db/DDD/p232_025.wav" >}}     |    {{< playbutton src="speech/hardclip/7db/DDD/p257_015.wav" >}}     |    {{< playbutton src="speech/hardclip/7db/DDD/p257_024.wav" >}}     |
|  Informed  |  {{< playbutton src="speech/hardclip/7db/informed/p232_013.wav" >}}  |  {{< playbutton src="speech/hardclip/7db/informed/p232_025.wav" >}}  |  {{< playbutton src="speech/hardclip/7db/informed/p257_015.wav" >}}  |  {{< playbutton src="speech/hardclip/7db/informed/p257_024.wav" >}}  |
|    MLP     |    {{< playbutton src="speech/hardclip/7db/mlp/p232_013.wav" >}}     |    {{< playbutton src="speech/hardclip/7db/mlp/p232_025.wav" >}}     |    {{< playbutton src="speech/hardclip/7db/mlp/p257_015.wav" >}}     |    {{< playbutton src="speech/hardclip/7db/mlp/p257_024.wav" >}}     |
|    CCR     |    {{< playbutton src="speech/hardclip/7db/ccr/p232_013.wav" >}}     |    {{< playbutton src="speech/hardclip/7db/ccr/p232_025.wav" >}}     |    {{< playbutton src="speech/hardclip/7db/ccr/p257_015.wav" >}}     |    {{< playbutton src="speech/hardclip/7db/ccr/p257_024.wav" >}}     |


## Softclipping

# 1dB

|            |                                S1  1                                 |                                 S1 2                                 |                                 S2 1                                 |                                 S2 2                                 |
|:----------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|
|   Clean    |  {{< playbutton src="speech/softclip/1db/original/p232_013.wav" >}}  |  {{< playbutton src="speech/softclip/1db/original/p232_025.wav" >}}  |  {{< playbutton src="speech/softclip/1db/original/p257_015.wav" >}}  |  {{< playbutton src="speech/softclip/1db/original/p257_024.wav" >}}  |
|  Clipped   |  {{< playbutton src="speech/softclip/1db/clipped/p232_013.wav" >}}  | {{< playbutton src="speech/softclip/1db/clipped/p232_025.wav" >}}  | {{< playbutton src="speech/softclip/1db/clipped/p257_015.wav" >}}  | {{< playbutton src="speech/softclip/1db/clipped/p257_024.wav" >}}  |
|  Resemble  |  {{< playbutton src="speech/softclip/1db/resemble/p232_013.wav" >}}  |  {{< playbutton src="speech/softclip/1db/resemble/p232_025.wav" >}}  |  {{< playbutton src="speech/softclip/1db/resemble/p257_015.wav" >}}  |  {{< playbutton src="speech/softclip/1db/resemble/p257_024.wav" >}}  |
| Voicefixer | {{< playbutton src="speech/softclip/1db/voicefixer/p232_013.wav" >}} | {{< playbutton src="speech/softclip/1db/voicefixer/p232_025.wav" >}} | {{< playbutton src="speech/softclip/1db/voicefixer/p257_015.wav" >}} | {{< playbutton src="speech/softclip/1db/voicefixer/p257_024.wav" >}} |
|    DDD     |    {{< playbutton src="speech/softclip/1db/DDD/p232_013.wav" >}}     |    {{< playbutton src="speech/softclip/1db/DDD/p232_025.wav" >}}     |    {{< playbutton src="speech/softclip/1db/DDD/p257_015.wav" >}}     |    {{< playbutton src="speech/softclip/1db/DDD/p257_024.wav" >}}     |
|  Informed  |  {{< playbutton src="speech/softclip/1db/informed/p232_013.wav" >}}  |  {{< playbutton src="speech/softclip/1db/informed/p232_025.wav" >}}  |  {{< playbutton src="speech/softclip/1db/informed/p257_015.wav" >}}  |  {{< playbutton src="speech/softclip/1db/informed/p257_024.wav" >}}  |
|    MLP     |    {{< playbutton src="speech/softclip/1db/mlp/p232_013.wav" >}}     |    {{< playbutton src="speech/softclip/1db/mlp/p232_025.wav" >}}     |    {{< playbutton src="speech/softclip/1db/mlp/p257_015.wav" >}}     |    {{< playbutton src="speech/softclip/1db/mlp/p257_024.wav" >}}     |
|    CCR     |    {{< playbutton src="speech/softclip/1db/ccr/p232_013.wav" >}}     |    {{< playbutton src="speech/softclip/1db/ccr/p232_025.wav" >}}     |    {{< playbutton src="speech/softclip/1db/ccr/p257_015.wav" >}}     |    {{< playbutton src="speech/softclip/1db/ccr/p257_024.wav" >}}     |


### 3dB

|            |                                S1  1                                 |                                 S1 2                                 |                                 S2 1                                 |                                 S2 2                                 |
|:----------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|
|   Clean    |  {{< playbutton src="speech/softclip/3db/original/p232_013.wav" >}}  |  {{< playbutton src="speech/softclip/3db/original/p232_025.wav" >}}  |  {{< playbutton src="speech/softclip/3db/original/p257_015.wav" >}}  |  {{< playbutton src="speech/softclip/3db/original/p257_024.wav" >}}  |
|  Clipped   |  {{< playbutton src="speech/softclip/3db/clipped/p232_013.wav" >}}  | {{< playbutton src="speech/softclip/3db/clipped/p232_025.wav" >}}  | {{< playbutton src="speech/softclip/3db/clipped/p257_015.wav" >}}  | {{< playbutton src="speech/softclip/3db/clipped/p257_024.wav" >}}  |
|  Resemble  |  {{< playbutton src="speech/softclip/3db/resemble/p232_013.wav" >}}  |  {{< playbutton src="speech/softclip/3db/resemble/p232_025.wav" >}}  |  {{< playbutton src="speech/softclip/3db/resemble/p257_015.wav" >}}  |  {{< playbutton src="speech/softclip/3db/resemble/p257_024.wav" >}}  |
| Voicefixer | {{< playbutton src="speech/softclip/3db/voicefixer/p232_013.wav" >}} | {{< playbutton src="speech/softclip/3db/voicefixer/p232_025.wav" >}} | {{< playbutton src="speech/softclip/3db/voicefixer/p257_015.wav" >}} | {{< playbutton src="speech/softclip/3db/voicefixer/p257_024.wav" >}} |
|    DDD     |    {{< playbutton src="speech/softclip/3db/DDD/p232_013.wav" >}}     |    {{< playbutton src="speech/softclip/3db/DDD/p232_025.wav" >}}     |    {{< playbutton src="speech/softclip/3db/DDD/p257_015.wav" >}}     |    {{< playbutton src="speech/softclip/3db/DDD/p257_024.wav" >}}     |
|  Informed  |  {{< playbutton src="speech/softclip/3db/informed/p232_013.wav" >}}  |  {{< playbutton src="speech/softclip/3db/informed/p232_025.wav" >}}  |  {{< playbutton src="speech/softclip/3db/informed/p257_015.wav" >}}  |  {{< playbutton src="speech/softclip/3db/informed/p257_024.wav" >}}  |
|    MLP     |    {{< playbutton src="speech/softclip/3db/mlp/p232_013.wav" >}}     |    {{< playbutton src="speech/softclip/3db/mlp/p232_025.wav" >}}     |    {{< playbutton src="speech/softclip/3db/mlp/p257_015.wav" >}}     |    {{< playbutton src="speech/softclip/3db/mlp/p257_024.wav" >}}     |
|    CCR     |    {{< playbutton src="speech/softclip/3db/ccr/p232_013.wav" >}}     |    {{< playbutton src="speech/softclip/3db/ccr/p232_025.wav" >}}     |    {{< playbutton src="speech/softclip/3db/ccr/p257_015.wav" >}}     |    {{< playbutton src="speech/softclip/3db/ccr/p257_024.wav" >}}     |

### 7dB

|            |                                S1  1                                 |                                 S1 2                                 |                                 S2 1                                 |                                 S2 2                                 |
|:----------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|:--------------------------------------------------------------------:|
|   Clean    |  {{< playbutton src="speech/softclip/7db/original/p232_013.wav" >}}  |  {{< playbutton src="speech/softclip/7db/original/p232_025.wav" >}}  |  {{< playbutton src="speech/softclip/7db/original/p257_015.wav" >}}  |  {{< playbutton src="speech/softclip/7db/original/p257_024.wav" >}}  |
|  Clipped   |  {{< playbutton src="speech/softclip/7db/clipped/p232_013.wav" >}}  | {{< playbutton src="speech/softclip/7db/clipped/p232_025.wav" >}}  | {{< playbutton src="speech/softclip/7db/clipped/p257_015.wav" >}}  | {{< playbutton src="speech/softclip/7db/clipped/p257_024.wav" >}}  |
|  Resemble  |  {{< playbutton src="speech/softclip/7db/resemble/p232_013.wav" >}}  |  {{< playbutton src="speech/softclip/7db/resemble/p232_025.wav" >}}  |  {{< playbutton src="speech/softclip/7db/resemble/p257_015.wav" >}}  |  {{< playbutton src="speech/softclip/7db/resemble/p257_024.wav" >}}  |
| Voicefixer | {{< playbutton src="speech/softclip/7db/voicefixer/p232_013.wav" >}} | {{< playbutton src="speech/softclip/7db/voicefixer/p232_025.wav" >}} | {{< playbutton src="speech/softclip/7db/voicefixer/p257_015.wav" >}} | {{< playbutton src="speech/softclip/7db/voicefixer/p257_024.wav" >}} |
|    DDD     |    {{< playbutton src="speech/softclip/7db/DDD/p232_013.wav" >}}     |    {{< playbutton src="speech/softclip/7db/DDD/p232_025.wav" >}}     |    {{< playbutton src="speech/softclip/7db/DDD/p257_015.wav" >}}     |    {{< playbutton src="speech/softclip/7db/DDD/p257_024.wav" >}}     |
|  Informed  |  {{< playbutton src="speech/softclip/7db/informed/p232_013.wav" >}}  |  {{< playbutton src="speech/softclip/7db/informed/p232_025.wav" >}}  |  {{< playbutton src="speech/softclip/7db/informed/p257_015.wav" >}}  |  {{< playbutton src="speech/softclip/7db/informed/p257_024.wav" >}}  |
|    MLP     |    {{< playbutton src="speech/softclip/7db/mlp/p232_013.wav" >}}     |    {{< playbutton src="speech/softclip/7db/mlp/p232_025.wav" >}}     |    {{< playbutton src="speech/softclip/7db/mlp/p257_015.wav" >}}     |    {{< playbutton src="speech/softclip/7db/mlp/p257_024.wav" >}}     |
|    CCR     |    {{< playbutton src="speech/softclip/7db/ccr/p232_013.wav" >}}     |    {{< playbutton src="speech/softclip/7db/ccr/p232_025.wav" >}}     |    {{< playbutton src="speech/softclip/7db/ccr/p257_015.wav" >}}     |    {{< playbutton src="speech/softclip/7db/ccr/p257_024.wav" >}}     |


