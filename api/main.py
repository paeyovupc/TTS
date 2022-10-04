import io
import os
import tempfile
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer


class TTSModel(BaseModel):
    language: str
    dataset: str
    model_name: str
    text: str


# update in-use models to the specified released models.
model_path = None
config_path = None
speakers_file_path = None
vocoder_path = None
vocoder_config_path = None

models_config_path = os.path.join(os.path.dirname(__file__).replace("api", "TTS"), ".models.json")
manager = ModelManager(models_config_path)
models_list = [
    {"language": model.split("/")[1], "dataset": model.split("/")[2], "model_name": model.split("/")[3]}
    for model in manager.list_models()
    if model.startswith("tts_models")
]

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/all_models")
async def get_all_models():
    return models_list


@app.get("/tts")
async def get_tts_audio(tts: TTSModel):
    print(" > Dataset: {}".format(tts.dataset))
    print(" > Language: {}".format(tts.language))
    print(" > Model name: {}".format(tts.model_name))
    print(" > Text: {}".format(tts.text))

    # TODO: check TTS/bin/synthesize for other options
    model_name = "tts_models/{}/{}/{}".format(tts.language, tts.dataset, tts.model_name)
    model_path, config_path, model_item = manager.download_model(model_name)
    vocoder_name = model_item["default_vocoder"]
    vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)
    
    # load models
    synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        tts_speakers_file=speakers_file_path,
        tts_languages_file=None,
        vocoder_checkpoint=vocoder_path,
        vocoder_config=vocoder_config_path,
        encoder_checkpoint="",
        encoder_config="",
        use_cuda=False,
    )

    wavs = synthesizer.tts(tts.text)
    tmp_dir = tempfile._get_default_tempdir()
    output_path = os.path.join(tmp_dir, datetime.now().strftime("tts_%m_%d_%Y_%H_%M_%S.wav"))
    synthesizer.save_wav(wavs, output_path)
    return FileResponse(output_path)
