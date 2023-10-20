# Speech Sequence Embeddings using Nearest Neighbors Contrastive Learning
This code can be used to reproduce the main results from [our paper](https://arxiv.org/abs/2204.05148) 

from Robin Algayres, Adel Nabli, Benoit Sagot, Emmanuel Dupoux
[Our website](https://cognitive-ml.fr/)

### Conda env equirements:
pip install numpy, torch==2.0, torchaudio==2.0, pytorch-metric-learning==0.9.95, fairseq==0.12.2, faiss-cpu==1.7.4

Please also install [augment](https://github.com/facebookresearch/WavAugment)

### This github contains
- pretrained Wav2vec2 model (small version) and pretrained SSE models (unsupervised and weakly supervised version)
- training files to reproduce our results on the LibriSpeech and Mandarin corpora

### Quick training of SSE model on LibriSpeech

Log on a machine with one GPU with 16Go of RAM and at least 20 cpus. Then launch the following commands.
- Get a path to the LibriSpeech corpus:
    - LS=\<path to LibriSpeech folder\> 
- Extract Wav2vec2.0 frames:
    - python utils/extract_features.py --path_wavs=$LS --path_vads=training_data/train-clean-360-subset-vads --output_dir=features/train-clean-360-subset/ 
    - python utils/extract_features.py --path_wavs=$LS --path_vads=training_data/dev-clean-vads --output_dir=features/dev-clean-subset
- Train the initial SSE model with data augmentation 
    - bash launch_init.sh $LS 
- Extract k-NN positive pairs and train a new SSE model
    - bash iteration.sh $LS checkpoints/\<model trained with previous script\> 
    - bash iteration.sh $LS checkpoints/\<model trained with previous script\>

- Optional: train a weakly supervised model: 
    - bash launch_gold.sh $LS

### Quick training of SSE model on Mandarin

Download the [mandarin wavs](https://zerospeech.com/tracks/2017/data/) from the Zerospeech website. Uncomment the Mandarin parameters in launch_init.sh and iteration.sh.
    - MAND=\<path to Mandarin folder\> 
    - python utils/extract_features.py --path_wavs=$MAND --path_vads=training_data/
    - bash launch_init.sh $MAND
    - bash iteration.sh $MAND checkpoints/\<model trained with previous script\> 
    - bash iteration.sh $MAND checkpoints/\<model trained with previous script\>

#### Evaluating and creating learned embeddings

Given list of labelled speech segments and a trained SSE model, you can compute a MAP score or embed the segments into a file.

- To get an MAP score: 
    - python map_dump.py --path_wavs=$LS --path_segments=sse_benchmark/dev-clean-ngrams-subset --task=map --path_sse=pretrained/librispeech_unsup/

- To embed a list of segments: 
    - python map_dump.py --path_wavs=$LS --path_segments=sse_benchmark/dev-clean-ngrams-subset --task=dump --path_sse=pretrained/librispeech_unsup/ --output_file=out

### Train a SSE model on a new corpus

Our unsupervised SSE model is trained by data augmentation of speech segments. The list of segment used in our paper can be found at training_data/train-clean-360-subset-segments. Each line in the list is built as follows: \
    \<path to wav\> \<spearker id\> \<vad start\> \<vad end\> \<segment start\> \<segment end\> \<transcription (optional)\>

To build the list of segments from speech dataset :

- Create a file containing all VAD (Voice Activity Dectection) sections found in the speech dataset. Each line in the list should be built as: 
\<path to wav\> \<spearker id\> \<vad start\> \<vad end\> \
We suggest using [pyannote](https://github.com/pyannote/pyannote-audio) to find the VAD sections (even though we used rVAD in the paper, pyannote works better)

- Extract the Wav2vec2 frames with
    - python utils/extract_features.py --path_wavs=<path to wavs> --path_vads=<vad file> --output_dir=<output feature file> 

- Write a the segment file using the vad file with the following script:
    - python utils/make_segments.py \<vad file\> \<output segment filename\>

- Modify laucher_init.sh and iterations.sh with the new vads and segments files as explained above.

- By default, models are trained until the MAP score computed on the validation set reaches the highest value. If you do not wish to use MAP for validation, you can add "--no_map" to the training command line. The training will be done in 5 epochs and should get close to the best performances.   
