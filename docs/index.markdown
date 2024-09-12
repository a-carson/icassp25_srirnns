---
layout: splash
classes:
  - wide
---
<h2 align="center">Interpolation filter design for sample rate indepndent RNNs</h2>
<p style="font-size: 0.75em" align="center">
Alistair Carson, Alec Wright and Stefan Bilbao</p>
<p style="font-size: 0.75em" align="center">
Welcome to the accompanying web-page for our ICASSP25 submission.</p>
<p style="font-size: 0.75em" align="center">
</p>




###### <b>Abstract</b>
<p style="font-size: 0.75em">
Recurrent neural networks (RNNs) are effective at emulating the non-linear, stateful behavior of analog guitar amplifiers and distortion effects. Unlike the case of direct circuit simulation, RNNs have a fixed sample rate encoded in their model weights, making the sample rate non-adjustable during inference. Recent work has proposed increasing the sample rate of RNNs at inference (oversampling) by increasing the feedback delay length in samples, using a fractional delay filter for non-integer conversions. Here, we investigate the task of lowering the sample rate at inference (undersampling), and propose using an extrapolation filter to approximate the required fractional signal advance. We consider two filter design methods and analyze the impact of filter order on audio quality. Our results show that the correct choice of filter can give high quality results for both oversampling and undersampling; however, in some cases the sample rate adjustment leads to unwanted artefacts in the output signal. We analyse these failure cases through linearised stability analysis, showing that they result from instability around a fixed point. This approach enables an informed prediction of suitable interpolation filters for a given RNN model before runtime. 
</p>


###### <b>Audio Examples</b>
<p style="font-size: 0.75em">
Coming soon...
</p>
