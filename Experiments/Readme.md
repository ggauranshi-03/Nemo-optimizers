## Before running the script download and format the data
# 1. Download simple text data (WikiText-2)
wget https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt

# 2. Convert to JSONL format (one line per document)
# We treat the whole file as one document for simplicity here
jq -n --rawfile text train.txt '{"text": $text}' > train_data.jsonl

# 3. Pre-process into NeMo Binary Format
# Note: You need the NeMo scripts. If you installed via pip, find the script location or clone the repo.
# This assumes you are in the NVIDIA NeMo container or have cloned the repo.
# If you don't have this script, you can clone it: git clone https://github.com/NVIDIA/NeMo.git
python NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input=train_data.jsonl \
    --json-keys=text \
    --tokenizer-library=huggingface \
    --tokenizer-type=gpt2 \
    --output-prefix=my_real_data \
    --append-eod \
    --workers=1

## For Yogi_vs_adam_realdata.py
Run 1: Train with Yogi
```
python Yogi_vs_adam_realdata.py --optimizer yogi --max_steps 500
```

Run 2: Train with AdamW
```
python your_script.py --optimizer adamw --max_steps 500
```