---
layout: splash
classes:
  - wide
---

<style>
        /* Flexbox container to align images side by side */
        .image-container {
            display: flex;
            justify-content: space-between; /* Adjust spacing between images */
        }

        /* Style for each figure element */
        figure {
            text-align: center;
            margin: 0 50px; /* Add some space between the images */
        }

        /* Ensure images are responsive */
        img {
            max-width: 100%; /* Makes sure the image doesn't overflow */
            height: auto;
        }

        /* Optional: Add caption styling */
        figcaption {
            font-style: italic;
            font-size: 0.75em;
            margin-top: 5px;
            text-align: center;
        }
</style>

<h2 style="font-size: 1.5em" align="center">Interpolation filter design for sample rate independent audio effect RNNs</h2>
<p style="font-size: 1.0em" align="center">
Alistair Carson, Alec Wright and Stefan Bilbao
</p>
<p style="font-size: 0.75em" align="center">
<i><a href="https://www.acoustics.ed.ac.uk/" target="_blank" rel="noopener noreferrer">Acoustics and Audio Group</a><br>University of Edinburgh</i> <br>Edinburgh, UK
</p>
<p style="font-size: 1.0em; text-align: center">
Welcome to the accompanying web-page for our ICASSP '25 submission.</p>
<div style="text-align: center; align-items: center">
    <a href="https://arxiv.org/abs/2409.15884" 
        class="btn btn--primary btn--small"
        target="_blank" rel="noopener noreferrer">
    🗞️ Paper
    </a>
    <a href="https://github.com/a-carson/icassp25_srirnns" 
        class="btn btn--primary btn--small"
        target="_blank" rel="noopener noreferrer">
    </> Code
    </a>
</div>

##### Abstract
<p style="font-size: 0.75em">
Recurrent neural networks (RNNs) are effective at emulating the non-linear, stateful behavior of analog guitar amplifiers and distortion effects. Unlike the case of direct circuit simulation, RNNs have a fixed sample rate encoded in their model weights, making the sample rate non-adjustable during inference. Recent work has proposed increasing the sample rate of RNNs at inference (oversampling) by increasing the feedback delay length in samples, using a fractional delay filter for non-integer conversions. Here, we investigate the task of lowering the sample rate at inference (undersampling), and propose using an extrapolation filter to approximate the required fractional signal advance. We consider two filter design methods and analyze the impact of filter order on audio quality. Our results show that the correct choice of filter can give high quality results for both oversampling and undersampling; however, in some cases the sample rate adjustment leads to unwanted artefacts in the output signal. We analyse these failure cases through linearised stability analysis, showing that they result from instability around a fixed point. This approach enables an informed prediction of suitable interpolation filters for a given RNN model before runtime. 
</p>

#### Filter designs
<p style="font-size: 0.75em">
Below are the fractional delay filter designs used in the sample rate independent RNNs for a) oversampling by a non-integer factor and b) undersampling by the inverse ratio. 
The top row shows the magnitude response in dB and the bottom row shows phase delay error in samples. 
</p>
<div class="image-container">
    <figure>
        <img src="img/kernels_up.png" alt="Image 1 description">
        <figcaption>a) Oversampling 44.1kHz -> 48kHz </figcaption>
    </figure>
    <figure>
        <img src="img/kernels_down.png" alt="Image 2 description">
        <figcaption >b) Undersampling 44.1kHz -> 40.5kHz</figcaption>
    </figure>
</div>


#### Audio Examples

<p style="font-size: 0.75em">
Below are audio examples from a selection of amplifier/effects models from the GuitarML Tone Library. <a href="https://guitarml.com/tonelibrary/tonelib-pro.html" 
target="_blank" rel="noopener noreferrer"> Click here to download all models </a>.
<br>
<br>
🎧 Headphones recommended to hear the differences 🎧
<br>
<br>
❗ WARNING: KEEP VOLUME LOW (some clips contain high frequency ringing artefacts) ❗
</p>

###### 1) Blackstar HT40 tube amp -- high gain

<p style="font-size: 0.75em">
    <br>
    Model path: <code>Proteus_Tone_Packs/AmpPack1/BlackstarHT40_AmpHighGain.json</code>
</p>
<div class="image-container">
<figure style="width:20%">
    <img src="img/ht-club-40-mkll-6l6-amp-front.jpg" alt="Blackstar HT40">
    <figcaption style="font-size: 0.5em">Image for reference only - not the actual device used to train the model! Source: <a href="https://blackstaramps.com/product/ht-club-40-6l6-mkii/" target="_blank" rel="noopener noreferrer">https://blackstaramps.com/product/ht-club-40-6l6-mkii/</a></figcaption>
</figure>
</div>
<br>
<table>
  <thead>
    <tr>
      <th colspan="1" style="text-align: center; background: white"></th>
      <th colspan="2" style="text-align: center">Clip 1</th>
      <th colspan="2" style="text-align: center">Clip 2</th>
      <th colspan="2" style="text-align: center">Clip 3</th>
    </tr>
    <tr>
      <th colspan="1" style="background: white; text-align: center; font-weight: bold">Input </th>
      <th colspan="2" style="background: white; text-align: center;">
        <audio controls style="width: 9em">
          <source src="audio/clip-4.wav" type="audio/wav">
        </audio></th>
      <th colspan="2" style="background: white; text-align: center;">
        <audio controls style="width: 9em">
          <source src="audio/clip-5.wav" type="audio/wav">
        </audio></th>
      <th colspan="2" style="background: white; text-align: center;">
        <audio controls style="width: 9em">
          <source src="audio/clip-6.wav" type="audio/wav">
        </audio></th>
    </tr>
    <tr>
      <th colspan="1" style="background: white; text-align: center; font-weight: bold">Target </th>
      <th colspan="2" style="background: white; text-align: center;">
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_base.wav" type="audio/wav">
        </audio></th>
      <th colspan="2" style="background: white; text-align: center;">
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-5_base.wav" type="audio/wav">
        </audio></th>
      <th colspan="2" style="background: white; text-align: center;">
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-6_base.wav" type="audio/wav">
        </audio></th>
    </tr>
    <tr>
      <th colspan="1" style="text-align: center"></th>
      <th colspan="6" style="text-align: center">Inference sample rate</th>
    </tr>
    <tr>
      <th style="text-align: center">Method</th>
      <th style="text-align: center">40.5kHz </th>
      <th style="text-align: center">48kHz </th>
      <th style="text-align: center">40.5kHz </th>
      <th style="text-align: center">48kHz </th>
      <th style="text-align: center">40.5kHz </th>
      <th style="text-align: center">48kHz </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center">Naive</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_naive.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_naive.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-5_naive.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_naive.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-6_naive.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_naive.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-1</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_L1.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-2</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_L2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_L2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_L2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_L2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_L2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_L2.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-3</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_L3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_L3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_L3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_L3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_L3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_L3.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-4</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_L4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_L4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_L4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_L4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_L4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_L4.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-5</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_L5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_L5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_L5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_L5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_L5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_L5.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-1</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_M1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_M1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_M1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_M1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_M1.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-2</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_M2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_M2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_M2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_M2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_M2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_M2.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-3</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_M3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_M3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_M3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_M3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_M3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_M3.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-4</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_M4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_M4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_M4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_M4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_M4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_M4.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-5</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_M5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_M5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_M5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-5_M5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_M5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-6_M5.wav" type="audio/wav">
        </audio></td>
    </tr>
  </tbody>
</table>
<br>

###### 2) Blues Junior -- clean

<p style="font-size: 0.75em">
<br>
Model path: <code>Proteus_Tone_Packs/AmpPack1/BluesJrAmp_VolKnob.json</code> 
</p>
<div class="image-container">
<figure style="width: 20%">
    <img src="img/0213205700_amp_frt_001_nr.jpg" alt="Blues Junior">
    <figcaption style="font-size: 0.5em">Image for reference only - not the actual device used to train the model! Source: <a href="https://www.fender.com/en-GB/guitar-amplifiers/vintage-pro-tube/blues-junior-lacquered-tweed/0213245700.html" target="_blank" rel="noopener noreferrer">https://www.fender.com/en-GB/guitar-amplifiers/vintage-pro-tube/blues-junior-lacquered-tweed/0213245700.html</a></figcaption>
</figure>
</div>
<br>
<table>
  <thead>
    <tr>
      <th colspan="1" style="text-align: center; background: white"></th>
      <th colspan="2" style="text-align: center">Clip 1</th>
      <th colspan="2" style="text-align: center">Clip 2</th>
      <th colspan="2" style="text-align: center">Clip 3</th>
    </tr>
    <tr>
      <th colspan="1" style="background: white; text-align: center; font-weight: bold">Input</th>
      <th colspan="2" style="background: white; text-align: center;">
        <audio controls style="width: 9em">
          <source src="audio/clip-4.wav" type="audio/wav">
        </audio></th>
      <th colspan="2" style="background: white; text-align: center;">
        <audio controls style="width: 9em">
          <source src="audio/clip-5.wav" type="audio/wav">
        </audio></th>
      <th colspan="2" style="background: white; text-align: center;">
        <audio controls style="width: 9em">
          <source src="audio/clip-6.wav" type="audio/wav">
        </audio></th>
    </tr>
    <tr>
      <th colspan="1" style="background: white; text-align: center; font-weight: bold">Target </th>
      <th colspan="2" style="background: white; text-align: center;">
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_down_clip-4_base.wav" type="audio/wav">
        </audio></th>
      <th colspan="2" style="background: white; text-align: center;">
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_down_clip-5_base.wav" type="audio/wav">
        </audio></th>
      <th colspan="2" style="background: white; text-align: center;">
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_down_clip-6_base.wav" type="audio/wav">
        </audio></th>
    </tr>
    <tr>
      <th colspan="1" style="text-align: center"></th>
      <th colspan="6" style="text-align: center">Inference sample rate</th>
    </tr>
    <tr>
      <th style="text-align: center">Method</th>
      <th style="text-align: center">40.5kHz </th>
      <th style="text-align: center">48kHz </th>
      <th style="text-align: center">40.5kHz </th>
      <th style="text-align: center">48kHz </th>
      <th style="text-align: center">40.5kHz </th>
      <th style="text-align: center">48kHz </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center">Naive</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_down_clip-4_naive.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-4_naive.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_down_clip-5_naive.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_naive.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_down_clip-6_naive.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_naive.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-1</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_down_clip-4_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-4_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_L1.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-2</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_down_clip-4_L2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-4_L2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_L2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_L2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_L2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_L2.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-3</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_down_clip-4_L3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-4_L3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_L3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_L3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_L3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_L3.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-4</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_down_clip-4_L4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-4_L4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_L4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_L4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_L4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_L4.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-5</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_down_clip-4_L5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-4_L5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_L5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_L5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_L5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_L5.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-1</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_down_clip-4_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-4_M1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_M1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_M1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_M1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_M1.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-2</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_down_clip-4_M2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-4_M2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_M2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_M2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_M2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_M2.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-3</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_down_clip-4_M3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-4_M3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_M3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_M3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_M3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_M3.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-4</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_down_clip-4_M4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-4_M4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_M4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_M4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_M4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_M4.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-5</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_down_clip-4_M5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-4_M5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_M5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-5_M5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_M5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/BluesJrAmp_VolKnob_up_clip-6_M5.wav" type="audio/wav">
        </audio></td>
    </tr>
  </tbody>
</table>
<br>



###### 3) Dumble Kit -- high gain

<p style="font-size: 0.75em">
<br>
Model path: <code>Proteus_Tone_Packs/AmpPack1/DumbleKit_HighG_DirectOut.json</code>
</p>

<div class="image-container">
<figure style="width:20%">
<img src="img/dumble-overdrive-special-00-50w-0020-5.jpg" alt="Dumble">
  <figcaption style="font-size: 0.5em">Image for reference only - not the actual device used to train the model! Source: <a href="https://www.dreamguitars.com/shop/amplification/amplifiers/dumble/dumble-overdrive-special-00-50w-0020/" target="_blank" rel="noopener noreferrer">https://www.dreamguitars.com/shop/amplification/amplifiers/dumble/dumble-overdrive-special-00-50w-0020/</a></figcaption>
</figure>
</div>
<br>
<table>
  <thead>
    <tr>
      <th colspan="1" style="text-align: center; background: white"></th>
      <th colspan="2" style="text-align: center">Clip 1</th>
      <th colspan="2" style="text-align: center">Clip 2</th>
      <th colspan="2" style="text-align: center">Clip 3</th>
    </tr>
    <tr>
      <th colspan="1" style="background: white; text-align: center; font-weight: bold">Input </th>
      <th colspan="2" style="background: white; text-align: center;">
        <audio controls style="width: 9em">
          <source src="audio/clip-4.wav" type="audio/wav">
        </audio></th>
      <th colspan="2" style="background: white; text-align: center;">
        <audio controls style="width: 9em">
          <source src="audio/clip-5.wav" type="audio/wav">
        </audio></th>
      <th colspan="2" style="background: white; text-align: center;">
        <audio controls style="width: 9em">
          <source src="audio/clip-6.wav" type="audio/wav">
        </audio></th>
    </tr>
    <tr>
      <th colspan="1" style="background: white; text-align: center; font-weight: bold">Target </th>
      <th colspan="2" style="background: white; text-align: center;">
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-4_base.wav" type="audio/wav">
        </audio></th>
      <th colspan="2" style="background: white; text-align: center;">
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-5_base.wav" type="audio/wav">
        </audio></th>
      <th colspan="2" style="background: white; text-align: center;">
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-6_base.wav" type="audio/wav">
        </audio></th>
    </tr>
    <tr>
      <th colspan="1" style="text-align: center"></th>
      <th colspan="6" style="text-align: center">Inference sample rate</th>
    </tr>
    <tr>
      <th style="text-align: center">Method</th>
      <th style="text-align: center">40.5kHz </th>
      <th style="text-align: center">48kHz </th>
      <th style="text-align: center">40.5kHz </th>
      <th style="text-align: center">48kHz </th>
      <th style="text-align: center">40.5kHz </th>
      <th style="text-align: center">48kHz </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center">Naive</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-4_naive.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-4_naive.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-5_naive.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_naive.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-6_naive.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_naive.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-1</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-4_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-4_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_L1.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-2</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-4_L2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-4_L2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_L2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_L2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_L2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_L2.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-3</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-4_L3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-4_L3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_L3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_L3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_L3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_L3.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-4</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-4_L4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-4_L4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_L4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_L4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_L4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_L4.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-5</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-4_L5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-4_L5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_L5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_L5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_L5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_L5.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-1</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-4_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-4_M1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_M1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_M1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_M1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_M1.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-2</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-4_M2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-4_M2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_M2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_M2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_M2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_M2.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-3</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-4_M3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-4_M3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_M3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_M3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_M3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_M3.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-4</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-4_M4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-4_M4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_M4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_M4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_M4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_M4.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-5</td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-4_M5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-4_M5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_M5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_M5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_M5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 9em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-6_M5.wav" type="audio/wav">
        </audio></td>
    </tr>
  </tbody>
</table>
<br>