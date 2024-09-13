---
layout: splash
classes:
  - wide
---
<h2 align="center">Interpolation filter design for sample rate independent audio effect RNNs</h2>
<p style="font-size: 0.75em" align="center">
Alistair Carson, Alec Wright and Stefan Bilbao
</p>
<p style="font-size: 0.75em" align="center"><i>Acoustics and Audio Group, University of Edinburgh</i></p>

<p style="font-size: 0.75em" align="center">
Welcome to the accompanying web-page for our ICASSP25 submission.</p>
<p style="font-size: 0.75em" align="center">
</p>




###### <b>Abstract</b>
<p style="font-size: 0.75em">
Recurrent neural networks (RNNs) are effective at emulating the non-linear, stateful behavior of analog guitar amplifiers and distortion effects. Unlike the case of direct circuit simulation, RNNs have a fixed sample rate encoded in their model weights, making the sample rate non-adjustable during inference. Recent work has proposed increasing the sample rate of RNNs at inference (oversampling) by increasing the feedback delay length in samples, using a fractional delay filter for non-integer conversions. Here, we investigate the task of lowering the sample rate at inference (undersampling), and propose using an extrapolation filter to approximate the required fractional signal advance. We consider two filter design methods and analyze the impact of filter order on audio quality. Our results show that the correct choice of filter can give high quality results for both oversampling and undersampling; however, in some cases the sample rate adjustment leads to unwanted artefacts in the output signal. We analyse these failure cases through linearised stability analysis, showing that they result from instability around a fixed point. This approach enables an informed prediction of suitable interpolation filters for a given RNN model before runtime. 
</p>


###### <b>Audio Examples</b>

1) Blackstar HT40 tube amp -- high gain
<table>
  <thead>
    <tr>
      <th colspan="2" style="background: white; text-align: center; font-weight: normal">Input (clean) </th>
      <th style="background: white; text-align: center;">
        <audio controls style="width: 12em">
          <source src="audio/clip-4.wav" type="audio/wav">
        </audio></th>
    </tr>
    <tr>
      <th colspan="2" style="background: white; text-align: center; font-weight: normal">Target (original RNN as trained at 44.1kHz) </th>
      <th colspan="1" style="background: white; text-align: center;">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_base.wav" type="audio/wav">
        </audio></th>
    </tr>
    <tr>
      <th colspan="1" style="text-align: center"></th>
      <th colspan="2" style="text-align: center">Inference sample rate</th>
    </tr>
    <tr>
      <th style="text-align: center">Method</th>
      <th style="text-align: center">40.5kHz </th>
      <th style="text-align: center">48kHz </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center">Naive</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_naive.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_naive.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-1</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_L1.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-2</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_L2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_L2.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-3</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_L3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_L3.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-4</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_L4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_L4.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-5</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_L5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_L5.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-1</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_M1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_M1.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-2</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_M2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_M2.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-3</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_M3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_M3.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-4</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_M4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_M4.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-5</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_down_clip-4_M5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_up_clip-4_M5.wav" type="audio/wav">
        </audio></td>
    </tr>
  </tbody>
</table>
<br>

2) Dumble Kit High Gain
<table>
  <thead>
    <tr>
      <th colspan="2" style="background: white; text-align: center; font-weight: normal">Input (clean) </th>
      <th style="background: white; text-align: center;">
        <audio controls style="width: 12em">
          <source src="audio/clip-5.wav" type="audio/wav">
        </audio></th>
    </tr>
    <tr>
      <th colspan="2" style="background: white; text-align: center; font-weight: normal">Target (original RNN as trained at 44.1kHz) </th>
      <th colspan="1" style="background: white; text-align: center;">
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-5_base.wav" type="audio/wav">
        </audio></th>
    </tr>
    <tr>
      <th colspan="1" style="text-align: center"></th>
      <th colspan="2" style="text-align: center">Inference sample rate</th>
    </tr>
    <tr>
      <th style="text-align: center">Method</th>
      <th style="text-align: center">40.5kHz </th>
      <th style="text-align: center">48kHz </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center">Naive</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-5_naive.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_naive.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-1</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-5_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_L1.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-2</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-5_L2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_L2.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-3</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-5_L3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_L3.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-4</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-5_L4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_L4.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-5</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-5_L5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_L5.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-1</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-5_M1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_M1.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-2</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-5_M2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_M2.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-3</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-5_M3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_M3.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-4</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-5_M4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_M4.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-5</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_down_clip-5_M5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/DumbleKit_HighG_DirectOut_up_clip-5_M5.wav" type="audio/wav">
        </audio></td>
    </tr>
  </tbody>
</table>
<br>

3) ENGL Powerball E645 -- clean
<table>
  <thead>
    <tr>
      <th colspan="2" style="background: white; text-align: center; font-weight: normal">Input (clean) </th>
      <th style="background: white; text-align: center;">
        <audio controls style="width: 12em">
          <source src="audio/clip-6.wav" type="audio/wav">
        </audio></th>
    </tr>
    <tr>
      <th colspan="2" style="background: white; text-align: center; font-weight: normal">Target (original RNN as trained at 44.1kHz) </th>
      <th colspan="1" style="background: white; text-align: center;">
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_down_clip-6_base.wav" type="audio/wav">
        </audio></th>
    </tr>
    <tr>
      <th colspan="1" style="text-align: center"></th>
      <th colspan="2" style="text-align: center">Inference sample rate</th>
    </tr>
    <tr>
      <th style="text-align: center">Method</th>
      <th style="text-align: center">40.5kHz </th>
      <th style="text-align: center">48kHz </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center">Naive</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_down_clip-6_naive.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_up_clip-6_naive.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-1</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_down_clip-6_L1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_up_clip-6_L1.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-2</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_down_clip-6_L2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_up_clip-6_L2.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-3</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_down_clip-6_L3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_up_clip-6_L3.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-4</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_down_clip-6_L4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_up_clip-6_L4.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Lagrange-5</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_down_clip-6_L5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_up_clip-6_L5.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-1</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_down_clip-6_M1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_up_clip-6_M1.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-2</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_down_clip-6_M2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_up_clip-6_M2.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-3</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_down_clip-6_M3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_up_clip-6_M3.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-4</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_down_clip-6_M4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_up_clip-6_M4.wav" type="audio/wav">
        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">Minimax-5</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_down_clip-6_M5.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/ENGL_E645_Clean_EdoardoNapoli_up_clip-6_M5.wav" type="audio/wav">
        </audio></td>
    </tr>
  </tbody>
</table>
<br>