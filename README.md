

<h2 style="font-size: 1.5em" align="center">Interpolation filter design for sample rate independent audio effect RNNs</h2>
<p style="font-size: 1.0em" align="center">
Alistair Carson, Alec Wright and Stefan Bilbao
</p>
<p style="font-size: 0.75em" align="center">
<i><a href="https://www.acoustics.ed.ac.uk/" target="_blank" rel="noopener noreferrer">Acoustics and Audio Group</a><br>University of Edinburgh</i> <br>Edinburgh, UK
</p>
<p style="font-size: 1.0em; text-align: center">
Welcome to the accompanying code for our ICASSP '25 submission.</p>
<div style="text-align: center">
    <a href="https://a-carson.github.io/icassp25_srirnns/" 
        class="btn btn--primary btn--small"
        target="_blank" rel="noopener noreferrer">
    🔊 Audio Examples
    </a>
</div>


#### Requirements
Download the [Proteus Tone Library](https://github.com/GuitarML/ToneLibrary/releases/download/v1.0/Proteus_Tone_Packs.zip) and extract to the current directory.

Install code requirements:
```angular2html
conda env create -f conda_env.yaml
conda activate srirnn_jax
```

#### Example scripts

- Apply SRIRNN methods to a single LSTM model and view SNR results. Example usage:
```angular2html
python3 process_audio_and_get_snr.py -f Proteus_Tone_Packs/AmpPack1/BlackstarHT40_AmpHighGain.json
```

- Apply linear analysis to a single LSTM model to view pole locations with and without SRIRNN interpolation methods applied. Example usage:
```angular2html
python3 linear_analysis.py -f Proteus_Tone_Packs/AmpPack1/BlackstarHT40_AmpHighGain.json --method lagrange
```


