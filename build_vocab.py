from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from tqdm import tqdm


input_file = 'data/bpe_data.txt'
max_num_words = 8_000
model_type = 'bpe'
model_prefix = 'bpe.8k'
pad_id = 0
unk_id = 1
bos_id = 2
eos_id = 3

sentencepiece_params = ' '.join([
    '--input={}'.format(input_file),
    '--model_type={}'.format(model_type),
    '--model_prefix={}'.format(model_prefix),
    '--vocab_size={}'.format(max_num_words),
    '--pad_id={}'.format(pad_id),
    '--unk_id={}'.format(unk_id),
    '--bos_id={}'.format(bos_id),
    '--eos_id={}'.format(eos_id)
])
print(sentencepiece_params)
SentencePieceTrainer.Train(sentencepiece_params)
