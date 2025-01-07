---
date: '2025-01-02T13:33:03+01:00'
draft: false
title: 'Guitar evaluation'
summary: 'Hard/Soft clipping for speech signals versus baselines'
weight: 2
---


## Hard Clipping

### 1dB

{{< shared-audio-player  id="1">}}

| Method         |                                   1                                    |                                   2                                    |                                    3                                    |                                    4                                    |                                    5                                    | 
|:---------------|:----------------------------------------------------------------------:|:----------------------------------------------------------------------:|:-----------------------------------------------------------------------:|:-----------------------------------------------------------------------:|:-----------------------------------------------------------------------:| 
| **Clean**      |    {{< playbutton src="/guitar/hardclip/onedb/original/x_1.wav" >}}    |    {{< playbutton src="/guitar/hardclip/onedb/original/x_7.wav" >}}    |    {{< playbutton src="/guitar/hardclip/onedb/original/x_22.wav" >}}    |    {{< playbutton src="/guitar/hardclip/onedb/original/x_31.wav" >}}    |    {{< playbutton src="/guitar/hardclip/onedb/original/x_41.wav" >}}    |
| **Clipped**    |    {{< playbutton src="/guitar/hardclip/onedb/degraded/y_1.wav" >}}    |    {{< playbutton src="/guitar/hardclip/onedb/degraded/y_7.wav">}}     |    {{< playbutton src="/guitar/hardclip/onedb/degraded/y_22.wav">}}     |    {{< playbutton src="/guitar/hardclip/onedb/degraded/y_31.wav">}}     |    {{< playbutton src="/guitar/hardclip/onedb/degraded/y_41.wav">}}     |
| **Informed**   |  {{< playbutton src="/guitar/hardclip/onedb/informed/x_hat_1.wav" >}}  |  {{< playbutton src="/guitar/hardclip/onedb/informed/x_hat_7.wav" >}}  |  {{< playbutton src="/guitar/hardclip/onedb/informed/x_hat_22.wav" >}}  |  {{< playbutton src="/guitar/hardclip/onedb/informed/x_hat_31.wav" >}}  |  {{< playbutton src="/guitar/hardclip/onedb/informed/x_hat_41.wav" >}}  |
| **Supervised** | {{< playbutton src="/guitar/hardclip/onedb/supervised/x_hat_1.wav" >}} | {{< playbutton src="/guitar/hardclip/onedb/supervised/x_hat_7.wav" >}} | {{< playbutton src="/guitar/hardclip/onedb/supervised/x_hat_22.wav" >}} | {{< playbutton src="/guitar/hardclip/onedb/supervised/x_hat_31.wav" >}} | {{< playbutton src="/guitar/hardclip/onedb/supervised/x_hat_41.wav" >}} |
| **SumTanh**    |  {{< playbutton src="/guitar/hardclip/onedb/sumtanh/x_hat_1.wav" >}}   |  {{< playbutton src="/guitar/hardclip/onedb/sumtanh/x_hat_7.wav" >}}   |  {{< playbutton src="/guitar/hardclip/onedb/sumtanh/x_hat_22.wav" >}}   |  {{< playbutton src="/guitar/hardclip/onedb/sumtanh/x_hat_31.wav" >}}   |  {{< playbutton src="/guitar/hardclip/onedb/sumtanh/x_hat_41.wav" >}}   |
| **MLP**        |    {{< playbutton src="/guitar/hardclip/onedb/mlp/x_hat_1.wav" >}}     |    {{< playbutton src="/guitar/hardclip/onedb/mlp/x_hat_7.wav" >}}     |    {{< playbutton src="/guitar/hardclip/onedb/mlp/x_hat_22.wav" >}}     |    {{< playbutton src="/guitar/hardclip/onedb/mlp/x_hat_31.wav" >}}     |    {{< playbutton src="/guitar/hardclip/onedb/mlp/x_hat_41.wav" >}}     |
| **CCR**        |    {{< playbutton src="/guitar/hardclip/onedb/ccr/x_hat_1.wav" >}}     |    {{< playbutton src="/guitar/hardclip/onedb/ccr/x_hat_7.wav" >}}     |    {{< playbutton src="/guitar/hardclip/onedb/ccr/x_hat_22.wav" >}}     |    {{< playbutton src="/guitar/hardclip/onedb/ccr/x_hat_31.wav" >}}     |    {{< playbutton src="/guitar/hardclip/onedb/ccr/x_hat_41.wav" >}}     |



