# LLM Internal States Inversion

This repository contains the implementation for the paper **"Depth Gives a False Sense of Privacy: LLM Internal States Inversion"** (USENIX Security 2025).

## Repository Structure

```text
llm-internal-states-inversion/
├── data/                   # Preprocessing code and datasets
├── scripts/                # Example scripts to run attacks and reproduce main results
├── results/                # Pretrained models, inversion results and training logs
├── white-box-inversion/    # Optimization-based attacks
└── black-box-inversion/    # Replication and generation-based attacks
```

*Note*: This version does not include `./results` which includes our pretrained model and inversion results.
You can find it in the V1 of our repository on Zenodo.

## Setup

### Prerequisites & Installation

Our code should work for CUDA >= 12.1. It is recommended to test with GPUs of large VRAM (e.g., A6000 Ada).

For example, to install CUDA 12.8 in Ubuntu 24.04, you can follow the [instruction on NVIDIA webpage](https://developer.nvidia.com/cuda-12-8-0-download-archive):

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```

To install the dependencies, you can either use `pip` or `conda`:

```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
```

## Dataset Preparation

### 1. Dataset Preprocessing

1.1 You can find our provided `long-context-datasets.json` under the folder `data/validation-set`. To reproduce, run `preprocess.py` to select the longest prompts from the following datasets evaluated in the paper:
[Aimedical](https://huggingface.co/datasets/ruslanmv/ai-medical-dataset),[Mentalhealth](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations),[CodeparrotApps](https://huggingface.co/datasets/codeparrot/apps), [Evolcode](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1). You can modify `preprocess.py` and the function `get_list_invert_text` of `white-box-inversion/utils.py` to run your own data. We also provide the short prompt data `data/validation-set/chat_2m_common.json` adapted from prior work [output2prompt](https://github.com/collinzrj/output2prompt).

1.2 It is recommended to download the evaluated models and set paths in `white-box-inversion/utils.py`: [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct), [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) [t5-base](https://huggingface.co/google-t5/t5-base), [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5).

1.3 To test your own data, one of the simplesy way is to replace your texts in one of the dictionary value of `./data/validation-set/long-context-datasets.json`.
For example, you can replace your own texts with the texts of `mentalhealth` in `./data/validation-set/long-context-datasets.json` and run the following attack scripts with the dataset name `mentalhealth`.
You can also prepare your own named dataset in `.json` similarlty, and modify the function in `get_list_invert_text` of `./white-box-inversion/utils.py`.
Then you can attack with your own dataset name.

### 2. White-box Attack

The main scripts for white-box attack, `attack_batch.py` and `utils.py`, are under `./white-box-inversion/`.

**Run the attack**.
To evaluate our TBS attack, the script can be in the following format:

```bash
dataid=1 # the index of data you want to attack
layerid=16 # the layer index observed by the adversary
modelname=llama3-8binst # the model name specified in LLM_PATH of the ./white-box-inversion/utils.py
dataset=mentalhealth # the dataset name defined in `get_list_invert_text` of `./white-box-inversion/utils.py`
CUDA_VISIBLE_DEVICES=0 python attack_batch.py     \
    --model-name $modelname \
    --dataset $dataset \
    --range $dataid:$((dataid+1)) \
    --folder long-prompt-$modelname/$dataset  \
    --access-layer-id $layerid  \
    --attack tbs \
    --num-steps 50000 \
    --lr 0.0001 \
    --w-dm 0.0 \
    --in-state-loss mse \
    --device cuda \
    --optim AdamW
```

More example scripts can be found in `scripts/main-whitebox-experiments/tbs-examples.sh`.

To evaluate the TS, ER and TBS attacks, please refer to the description of `attack_batch.py` under `./white-box-inversion` and see the examples in `scripts/main-whitebox-experiments/short-prompt.sh` and `scripts/main-whitebox-experiments/short-prompt.sh` which are used in our evaluation.

After running the `attack_batch.py`, you can find the results under `{results}/{folder}`, where `{results}` is specified by `RESULT_PATH` in `utils.py`, `{folder}` is specified by the argument `--folder`.
You will find result folder with name like this `0:1-tbs-l16-mse-lr0.0005w10.0w20.0wd0.0-ep50000-atan:5`, which represents the TBS attack for the sample index from 0 to 1 on the layer 16, with learning rate 0.0005, zero L1, L2 and DM penalty, 50000 epcohs and atan scaled by 5 as variable change function (specified by the function `tbs_to_invert_vector_atan` in `utils.py`).

**Evaluate the attack**.
We provide our inversion data under `results/white-box` for both short promtps and long prompts.
To reproduce the main evaluation results, please refer to the `.ipynb` notebooks under `white-box-inversion`.

### 3. Black-box Attack

The main scripts for black-box attacks are under `./black-box-inversion`.

We mainly show how to train and evaluate our generation-based attack because of its better performance.
For the extended optimization-based attack, please check the `black-box-inversion/replication`, where the `gen-is-dataset.py` generates the training data for the replication training by `replicate.py` with `config.yaml`. After training, you can test our optimization attacks on the obtained model.

#### 3.1 Prepare the dataset

We use [HuggingFaceH4-no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) to prepare the dataset (stored under `$SAVE_FOLDER`) for inversion model training.
We have extracted the prompts into `data/black-box-inversion/processed-dataset/HuggingFaceH4-no_robots`
For example, the command to prepare dataset with Llama-3.1-8B-Instruct's middle layer is:

```bash
python prepare_dataset.py --model-name $PATH_TO_LLAMA31_8B_INSTRUCT --invert-model $PATH_TO_INVERT_MODEL  --dataset-name norobot-train  --access-layer-id 16 --maxseqlen 256 --savefolder $SAVE_FOLDER
```

For the test dataset, you do not need to set the `--maxseqlen`, e.g.:

```bash
python prepare_dataset.py --model-name $PATH_TO_LLAMA31_8B_INSTRUCT --dataset-name mentalhealth  --access-layer-id 16 --savefolder $SAVE_FOLDER
```

After that, you can see your dataset folder under `data/black-box-inversion`.
The commands are provided in `scripts/main-blackbox-experiments/example-prepare-dataset.sh` where you can set the variables `PATH_TO_LLAMA31_8B_INSTRUCT` and `PATH_TO_INVERT_MODEL` to obtain the training and test datasets for the inversion model.
For example, you can copy the bash scirpt under `./black-box-inversion` and run `CUDA_VISIBLE_DEVICES=0 bash example-prepare-dataset.sh`.

#### 3.2 Train the inversion model

You need to first set up a config file.
We provide the config file `black-box-inversion/config/llama3.1-8b.json` as an example for training inversion model on Llama-3.1-8B-Instruct's middle layer.
You need to set the `output_dir` to the path to save your model, `dataset_path` to the folder of the prepared training dataset in the first step (which should be `../data/black-box-inversion/long-prompt/inversion_norobot-train_t5-base_l16_Meta-Llama-3.1-8B-Instruct_msl256` if you follow the first step), and `model_name_or_path` to the path of T5-base model.
Then, run

```bash
CUDA_VISIBLE_DEVICES=0 python inversion.py --config ./config/llama3.1-8b.json
```

#### 3.3 Inference and Evaluation

To obtain inference results on the prepared test data in the first step, run for example

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --model-path ../results/black-box/pretrained-hs2output-norobot-l3.18binstruct-l16-msl256-ep10 --dataset-path ../data/black-box-inversion/long-prompt/inversion_mentalhealth_t5-base_l16_Meta-Llama-3.1-8B-Instruct --save-path ../results/black-box/eval/mentalhealth-l3.18binstruct-l16-msl256-ep10xxx.json
```

The inference results will be stored as a json file.
We have provided our pretrained inversion model with `--msl=256` and 10 epochs in `results/black-box/pretrained-hs2output-norobot-l3.18binstruct-l16-msl256-ep10`.
We also provided the inversion texts in `./results/black-box/eval`'.

To evaluate the inversion results, please check out `black-box-inversion/eval.ipynb`.

## License

## Acknowledgements

This project builds upon the work from [output2prompt](https://github.com/collinzrj/output2prompt) repository.
We gratefully acknowledge the original authors for their contribution.

## Citation

```latex
@inproceedings{dong2025internalinversion,
  author       = {Tian Dong and Yan Meng and Shaofeng Li and Guoxing Chen and Zhen Liu and Haojin Zhu},
  title = {Depth Gives a False Sense of Privacy: LLM Internal States Inversion},
  booktitle = {34th USENIX Security Symposium (USENIX Security 25)},
  publisher = {USENIX Association},
  year = {2025},
  month = aug
}
```
