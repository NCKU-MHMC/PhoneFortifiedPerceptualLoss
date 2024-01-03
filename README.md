# Phone Fortified Perceptual Loss for Speech Enhancement
This fork updates the project dependencies of the [official implementation](https://github.com/aleXiehta/PhoneFortifiedPerceptualLoss) of *"Improving Perceptual Quality by Phone-Fortified Perceptual Loss using Wasserstein Distance for Speech Enhancement"*, and makes minor modifications to the parts that cause errors due to out-of-date versions, so that they are sufficient to operate correctly.

## Requirements
```
poetry install
```

## Data preparation
#### Enhancement model parameters and the *wav2vec* pre-trained model
Please download the model weights from [here](https://drive.google.com/drive/folders/1cwDoGdF44ExQt__B6Z44g3opUdH-hJXE?usp=sharing), and put the weight file into the `PFPL-W` and `PFPL` folder, respectively.
The *wav2vec* pre-trained model can be found in the official [repo](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md#pre-trained-models-1).

#### Voice Bank--Demand Dataset
The Voice Bank--Demand Dataset is not provided by this repository. Please download the dataset and build your own PyTorch dataloader from [here](https://datashare.is.ed.ac.uk/handle/10283/1942?show=full).
For each `.wav` file, you need to first convert it into 16kHz format by any audio converter (e.g., [sox](http://sox.sourceforge.net/)).
```
sox <48K.wav> -r 16000 -c 1 -b 16 <16k.wav>
```

## Usage
#### Training
To train the model, please run the following script.
The full training process apporximately consumes 19GB of GPU vram. Reduce the batch size if needed.
```
python main.py \
    --exp_dir <root/dir/of/experiment> \
    --exp_name <name_of_the_experiment> \
    --data_dir <root/dir/of/dataset> \
    --num_workers 16 \
    --cuda \
    --log_interval 100 \
    --batch_size 28 \
    --learning_rate 0.0001 \
    --num_epochs 100 \
    --clip_grad_norm_val 0 \
    --grad_accumulate_batches 1 \
    --n_fft 512 \
    --hop_length 128 \
    --model_type wav2vec \
    --log_grad_norm
```
#### Testing
To generate the enhanced sound files, please run:
```
python generate.py <path/to/PFPL_or_PFPL-W> <path/to/output/dir>
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
* [Bio-ASP Lab](https://bio-asplab.citi.sinica.edu.tw), CITI, Academia Sinica, Taipei, Taiwan
