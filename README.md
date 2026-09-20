
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 scikit-learn==1.5.1 matplotlib==3.9.2
pip install transformers==4.56.1 accelerate==1.10.1 peft==0.17.1
# Install the PyTorch 2.8.0 build appropriate for the local CUDA environment:
# https://pytorch.org/get-started/locally/
```

## Running the classifiers

Run commands from the repository root. Use `classification2` for the binary task and `classification6` for the six-class task.

### Feature-based models

```bash
python SVM.py --data_dir classification2
python LR.py --data_dir classification2 --C 1.0
python NB.py --data_dir classification2 --alpha 0.1
```

To run the same implementations on the six-class dataset, change the data directory:

```bash
python SVM.py --data_dir classification6
```

### Neural classifiers

```bash
python textcnn.py --data_dir classification2 --save_dir outputs/textcnn_binary --fp16
python dpcnn.py --data_dir classification2 --save_dir outputs/dpcnn_binary --fp16
python BiLSTM-Attention.py --data_dir classification2 --save_dir outputs/bilstm_binary --fp16
```

The GloVe file is optional for `textcnn.py`, `dpcnn.py`, and `BiLSTM-Attention.py`. `BIGRU-Attention.py` expects the path supplied through `--glove_path` to exist; pass the path to a compatible 300-dimensional GloVe-format file when using that script.

### Pretrained encoders

The pretrained-encoder scripts default to local checkpoint paths from the original experimental environment. Supply either a local checkpoint directory or a compatible Hugging Face model identifier through `--model_name_or_path`.

```bash
python BERT.py \
  --data_dir classification2 \
  --model_name_or_path bert-base-uncased \
  --save_dir outputs/bert_binary \
  --fp16

python SCIBERT.py \
  --data_dir classification2 \
  --model_name_or_path allenai/scibert_scivocab_uncased \
  --save_dir outputs/scibert_binary \
  --fp16

python RoBERTa.py \
  --data_dir classification6 \
  --model_name_or_path FacebookAI/roberta-base \
  --save_dir outputs/roberta_sixclass \
  --batch_size 32 \
  --grad_accum 1 \
  --fp16

python DeBERTa.py \
  --data_dir classification6 \
  --model_name_or_path microsoft/deberta-v3-base \
  --save_dir outputs/deberta_sixclass \
  --fp16
```

The scripts print validation and test metrics, including accuracy, macro-F1, and class-specific precision, recall, and F1. Neural and pretrained-encoder scripts save their best validation checkpoint to the directory supplied through `--save_dir`.

## Experimental settings

The common settings used in the manuscript include:

- random seed: `42`;
- maximum input length for feature-trained neural networks and pretrained encoders: `128` tokens;
- pretrained-encoder learning rate: `2e-5`;
- pretrained-encoder weight decay: `0.01`;
- pretrained-encoder warm-up ratio: `0.06`;
- early-stopping patience: `2` epochs for pretrained encoders and `3` epochs for task-trained neural networks.

Individual defaults and additional options are available through:

```bash
python SCRIPT_NAME.py --help
```

The experiments were designed for a single NVIDIA GPU with 8 GB of memory. Memory requirements vary by model, batch size, sequence length, and precision setting.

## Data availability and third-party material

This repository distributes bibliographic metadata, sentence-level research annotations, dataset partitions, and implementation code. It does not redistribute the source full-text PDFs because the articles are third-party publications. The bibliographic records in `Data.xlsx` can be used to identify the source articles.

The classification files contain sentence excerpts paired with labels created for this study. Copyright in the source wording remains with the respective authors and publishers. Users should cite this repository and the original source articles where appropriate.

The current repository does not yet contain a repository-level licence file. Copyright in the code and author-generated annotations remains with the authors until an explicit licence is added.
