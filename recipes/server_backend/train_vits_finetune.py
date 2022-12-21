import os
import argparse

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

MODELS_PATH = os.path.expanduser('~/.local/share/tts')
RESTORE_PATH_MAP = {
    'ca': {
        'female': os.path.join(MODELS_PATH, 'tts_models--ca--upc_ca_ona--vits', 'model_file.pth'),
        'male': os.path.join(MODELS_PATH, 'tts_models--ca--upc_ca_pau--vits', 'model_file.pth')
    },
    'es': {
        'female': os.path.join(MODELS_PATH, 'tts_models--es--upc_es_mariajose--vits', 'model_file.pth'),
        'male': os.path.join(MODELS_PATH, 'tts_models--es--upc_es_antonio--vits', 'model_file.pth')
    }
}

# db_path, language, sample_rate, out_path, voice_type
parser = argparse.ArgumentParser()
parser.add_argument('DB_PATH', type=str, help='Path to the database')
parser.add_argument('LANGUAGE', type=str, help='Language of the database', choices=RESTORE_PATH_MAP.keys())
parser.add_argument('SAMPLE_RATE', type=int, help='Sample rate of the database')
parser.add_argument('OUT_PATH', type=str, help='Output path for the models')
parser.add_argument('VOICE_TYPE', type=str, help='Type of voice: male or female', choices=['female', 'male'])
args = parser.parse_args()

output_path = args.OUT_PATH
model_restore_path = RESTORE_PATH_MAP[args.LANGUAGE][args.VOICE_TYPE]

dataset_config = BaseDatasetConfig(
    formatter='pae_upc', meta_file_train='metadata.txt', path=args.DB_PATH
)

audio_config = VitsAudioConfig(
    sample_rate=args.SAMPLE_RATE,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None
)

config = VitsConfig(
    audio=audio_config,
    run_name='vits_' + args.DB_PATH.split('/')[-1] + '_finetune',
    batch_size=32,
    eval_batch_size=16,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner='catalan_cleaners',
    use_phonemes=True,
    phoneme_language=args.LANGUAGE,
    phoneme_cache_path=os.path.join(output_path, 'phoneme_cache'),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=True,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    cudnn_benchmark=False,

    # Fine tune part
    lr_gen=0.00001,
    lr_disc=0.00001,
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init model
model = Vits(config, ap, tokenizer, speaker_manager=None)

# init the trainer
trainer = Trainer(
    TrainerArgs(
        # Fine tune part
        restore_path=model_restore_path
    ),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()

