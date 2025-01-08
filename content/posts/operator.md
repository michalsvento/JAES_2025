---
date: '2025-01-02T14:09:58+01:00'
draft: False
title: 'Operator evaluation'
summary: 'Precision of approximating nonlinear memoryless function'
weight: 2
---

{{< shared-audio-player id="1" >}}


### Half-wave rectification


|          |                             1                             |                             2                             |                             3                              |                             4                              |                             5                              |
|:--------:|:---------------------------------------------------------:|:---------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|:----------------------------------------------------------:|
| Original |    {{< playbutton src="operator/original/x_1.wav" >}}     |    {{< playbutton src="operator/original/x_7.wav" >}}     |    {{< playbutton src="operator/original/x_22.wav" >}}     |    {{< playbutton src="operator/original/x_31.wav" >}}     |    {{< playbutton src="operator/original/x_41.wav" >}}     |
| Degraded |  {{< playbutton src="operator/hwr/degraded/y_1.wav" >}}   |  {{< playbutton src="operator/hwr/degraded/y_7.wav" >}}   |  {{< playbutton src="operator/hwr/degraded/y_22.wav" >}}   |  {{< playbutton src="operator/hwr/degraded/y_31.wav" >}}   |  {{< playbutton src="operator/hwr/degraded/y_41.wav" >}}   |
| SumTanh  | {{< playbutton src="operator/hwr/sumtanh/x_hat_1.wav" >}} | {{< playbutton src="operator/hwr/sumtanh/x_hat_7.wav" >}} | {{< playbutton src="operator/hwr/sumtanh/x_hat_22.wav" >}} | {{< playbutton src="operator/hwr/sumtanh/x_hat_31.wav" >}} | {{< playbutton src="operator/hwr/sumtanh/x_hat_41.wav" >}} |
|   MLP    |   {{< playbutton src="operator/hwr/mlp/x_hat_1.wav" >}}   |   {{< playbutton src="operator/hwr/mlp/x_hat_7.wav" >}}   |   {{< playbutton src="operator/hwr/mlp/x_hat_22.wav" >}}   |   {{< playbutton src="operator/hwr/mlp/x_hat_31.wav" >}}   |   {{< playbutton src="operator/hwr/mlp/x_hat_41.wav" >}}   |
|   CCR    |   {{< playbutton src="operator/hwr/ccr/x_hat_1.wav" >}}   |   {{< playbutton src="operator/hwr/ccr/x_hat_7.wav" >}}   |   {{< playbutton src="operator/hwr/ccr/x_hat_22.wav" >}}   |   {{< playbutton src="operator/hwr/ccr/x_hat_31.wav" >}}   |   {{< playbutton src="operator/hwr/ccr/x_hat_41.wav" >}}   |


### Quantization

|          |                              1                              |                              2                              |                              3                               |                              4                               |                              5                               |
|:--------:|:-----------------------------------------------------------:|:-----------------------------------------------------------:|:------------------------------------------------------------:|:------------------------------------------------------------:|:------------------------------------------------------------:|
| Original |     {{< playbutton src="operator/original/x_1.wav" >}}      |     {{< playbutton src="operator/original/x_7.wav" >}}      |     {{< playbutton src="operator/original/x_22.wav" >}}      |     {{< playbutton src="operator/original/x_31.wav" >}}      |     {{< playbutton src="operator/original/x_41.wav" >}}      |
| Degraded |  {{< playbutton src="operator/quant/degraded/y_1.wav" >}}   |  {{< playbutton src="operator/quant/degraded/y_7.wav" >}}   |  {{< playbutton src="operator/quant/degraded/y_22.wav" >}}   |  {{< playbutton src="operator/quant/degraded/y_31.wav" >}}   |  {{< playbutton src="operator/quant/degraded/y_41.wav" >}}   |
| SumTanh  | {{< playbutton src="operator/quant/sumtanh/x_hat_1.wav" >}} | {{< playbutton src="operator/quant/sumtanh/x_hat_7.wav" >}} | {{< playbutton src="operator/quant/sumtanh/x_hat_22.wav" >}} | {{< playbutton src="operator/quant/sumtanh/x_hat_31.wav" >}} | {{< playbutton src="operator/quant/sumtanh/x_hat_41.wav" >}} |
|   MLP    |   {{< playbutton src="operator/quant/mlp/x_hat_1.wav" >}}   |   {{< playbutton src="operator/quant/mlp/x_hat_7.wav" >}}   |   {{< playbutton src="operator/quant/mlp/x_hat_22.wav" >}}   |   {{< playbutton src="operator/quant/mlp/x_hat_31.wav" >}}   |   {{< playbutton src="operator/quant/mlp/x_hat_41.wav" >}}   |
|   CCR    |   {{< playbutton src="operator/quant/ccr/x_hat_1.wav" >}}   |   {{< playbutton src="operator/quant/ccr/x_hat_7.wav" >}}   |   {{< playbutton src="operator/quant/ccr/x_hat_22.wav" >}}   |   {{< playbutton src="operator/quant/ccr/x_hat_31.wav" >}}   |   {{< playbutton src="operator/quant/ccr/x_hat_41.wav" >}}   |

### Foldback distortion

|          |                               1                                |                               2                                |                                3                                |                                4                                |                                5                                |
|:--------:|:--------------------------------------------------------------:|:--------------------------------------------------------------:|:---------------------------------------------------------------:|:---------------------------------------------------------------:|:---------------------------------------------------------------:|
| Original |       {{< playbutton src="operator/original/x_1.wav" >}}       |       {{< playbutton src="operator/original/x_7.wav" >}}       |       {{< playbutton src="operator/original/x_22.wav" >}}       |       {{< playbutton src="operator/original/x_31.wav" >}}       |       {{< playbutton src="operator/original/x_41.wav" >}}       |
| Degraded |  {{< playbutton src="operator/foldback/degraded/y_1.wav" >}}   |  {{< playbutton src="operator/foldback/degraded/y_7.wav" >}}   |  {{< playbutton src="operator/foldback/degraded/y_22.wav" >}}   |  {{< playbutton src="operator/foldback/degraded/y_31.wav" >}}   |  {{< playbutton src="operator/foldback/degraded/y_41.wav" >}}   |
| SumTanh  | {{< playbutton src="operator/foldback/sumtanh/x_hat_1.wav" >}} | {{< playbutton src="operator/foldback/sumtanh/x_hat_7.wav" >}} | {{< playbutton src="operator/foldback/sumtanh/x_hat_22.wav" >}} | {{< playbutton src="operator/foldback/sumtanh/x_hat_31.wav" >}} | {{< playbutton src="operator/foldback/sumtanh/x_hat_41.wav" >}} |
|   MLP    |   {{< playbutton src="operator/foldback/mlp/x_hat_1.wav" >}}   |   {{< playbutton src="operator/foldback/mlp/x_hat_7.wav" >}}   |   {{< playbutton src="operator/foldback/mlp/x_hat_22.wav" >}}   |   {{< playbutton src="operator/foldback/mlp/x_hat_31.wav" >}}   |   {{< playbutton src="operator/foldback/mlp/x_hat_41.wav" >}}   |
|   CCR    |   {{< playbutton src="operator/foldback/ccr/x_hat_1.wav" >}}   |   {{< playbutton src="operator/foldback/ccr/x_hat_7.wav" >}}   |   {{< playbutton src="operator/foldback/ccr/x_hat_22.wav" >}}   |   {{< playbutton src="operator/foldback/ccr/x_hat_31.wav" >}}   |   {{< playbutton src="operator/foldback/ccr/x_hat_41.wav" >}}   |

### Carbon Microphone distortion

|          |                               1                                |                                2                                |
|:--------:|:---------------------------------------------------------------:|:---------------------------------------------------------------:|
| Original |  {{< playbutton src="operator/carbonmic/original/x_1.wav" >}}   |  {{< playbutton src="operator/carbonmic/original/x_2.wav" >}}   |
| Degraded |  {{< playbutton src="operator/carbonmic/degraded/y_1.wav" >}}   |  {{< playbutton src="operator/carbonmic/degraded/y_2.wav" >}}   |
| SumTanh  | {{< playbutton src="operator/carbonmic/sumtanh/x_hat_1.wav" >}} | {{< playbutton src="operator/carbonmic/sumtanh/x_hat_2.wav" >}} |
|   MLP    |   {{< playbutton src="operator/carbonmic/mlp/x_hat_1.wav" >}}   |   {{< playbutton src="operator/carbonmic/mlp/x_hat_2.wav" >}}   |
|   CCR    |   {{< playbutton src="operator/carbonmic/ccr/x_hat_1.wav" >}}   |   {{< playbutton src="operator/carbonmic/ccr/x_hat_2.wav" >}}   |


