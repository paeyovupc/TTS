import io
import json
import os
import subprocess
import tempfile
import zipfile
from datetime import datetime
from math import floor
from pathlib import Path

import aiofiles
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

tmp_dir = tempfile._get_default_tempdir()
root_path = os.path.dirname(__file__).split("TTS")[0]
users_config_path = os.path.join(root_path, "TTS", "api", "users_config.json")
users_path = os.path.join(root_path, "users")
databases_path = os.path.join(root_path, "databases")
models_config_path = os.path.join(root_path, "TTS", "TTS", ".models.json")
train_script = os.path.join(root_path, "TTS", "recipes", "server_backend", "train_vits.py")

manager = ModelManager(models_config_path)
models_list = [
    {"language": model.split("/")[1], "dataset": model.split("/")[2], "model_name": model.split("/")[3]}
    for model in manager.list_models()
    if model.startswith("tts_models")
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def write_json(new_data, action, user=None):
    jsonFile = open(users_config_path, "r")
    data = json.load(jsonFile)
    jsonFile.close()

    if action == 'new_user':
        data["users"].update(new_data)
    elif action == 'new_model':
        data["users"][user]["models"].update(new_data)

    jsonFile = open(users_config_path, "w+")
    jsonFile.write(json.dumps(data, indent=2))
    jsonFile.close()


def zip_files(db_name: str):
    db_path = os.path.join(databases_path, db_name)
    zip_filename = "{}.zip".format(db_name)
    s = io.BytesIO()
    zf = zipfile.ZipFile(s, "w")

    # Create folder inside zip
    zf.write(db_path, db_name)

    # Iterate over all the files in directory
    for folderName, subfolders, filenames in os.walk(db_path):
        for filename in filenames:
            filePath = os.path.join(folderName, filename)
            zf.write(filePath, os.path.join(db_name, filename))
    zf.close()

    # Grab ZIP file from in-memory, make response with correct MIME-type
    resp = Response(
        s.getvalue(),
        media_type="application/zip",
        headers={'Content-Disposition': f'attachment;filename={zip_filename}'},
    )

    return resp


async def unzip_files(file: UploadFile, user_name: str):
    db_name = file.filename.split('.')[0]
    user_path = os.path.join(users_path, user_name)
    zip_path = os.path.join(user_path, file.filename)
    db_path = os.path.join(user_path, db_name)

    # Async writing zip file to disk
    async with aiofiles.open(zip_path, 'wb') as out_file:
        content = await file.read()  # async read
        await out_file.write(content)  # async write

    # Unzip files and delete zip
    zip = zipfile.ZipFile(zip_path)
    zip.extractall(user_path)
    zip.close()
    os.remove(zip_path)

    # Create symlink /users/user_name/db_name <--> /databases/user_name__db_name
    os.symlink(db_path, os.path.join(databases_path, '{}__{}'.format(user_name, db_name)))
    return db_path


def get_users_models_dict(user: str):
    f = open(users_config_path)
    data = json.load(f)
    return data['users'][user]['models']


def train_model(db_path_str: str, language: str):
    # Get some useful information about the database
    db_path = Path(db_path_str)
    wav_file = next(db_path.glob('*.wav'), None)

    if wav_file is None:
        return None

    # Database/model info
    db_name = db_path.name
    user = db_path.parent.name
    db_path = db_path_str
    sample_rate = sf.info(wav_file).samplerate
    out_path = Path(users_path) / user / db_name

    # Add model training to queue
    # INFO: change `tsp python3 args ...` to `tsp sh -c "python3 args ..."` in
    # case the command does not work
    # usage: train_vits.py DB_PATH LANGUAGE SAMPLE_RATE OUT_PATH
    command = [
        str(a) for a in [
            'tsp', 'python3', train_script, db_path, language, sample_rate,
            out_path
        ]
    ]
    # Make sure that everything is a string
    
    envvars = {'CUDA_VISIBLE_DEVICES': '0'}

    proc = subprocess.Popen(command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            env=envvars)
    db_id = proc.stdout.readline().decode('ascii').strip()

    proc = subprocess.Popen(['tsp', '-s', db_id],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    db_status = proc.stdout.readline().decode('ascii').strip()

    # Save new model to json
    model_name = f"tts_models/{language}/{db_name}/vits"
    write_json({model_name: db_id},
               'new_model',
               user=user)

    return {
        model_name: {
        'status': db_status,
        'progress': '0'
        }
    }


def check_user_models(user: str):
    """
    Checks the status of the given user models, and returns the data in the
    following format:
    
    [{
        "<model_name_1>": {
            "status": "queued",
            "progress": "0"
        }
    }, {
        "<model_name_2>": {
            "status": "running",
            "progress": "31"
        }
    }, {
        "<model_name_3>": {
            "status": "finished",
            "progress": "100"
        }
    }]
    """
    user_models = get_users_models_dict(user)

    result = []
    for model_name, model_id in user_models.items():
        foo = {}
        foo[model_name] = {"status": "queued", "progress": "0"}

        proc = subprocess.Popen(['tsp', '-s', str(model_id)],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        foo[model_name]["status"] = proc.stdout.readline().decode(
            'ascii').strip()

        if foo[model_name]["status"] == "queued":
            foo[model_name]["progress"] = "0"

        elif foo[model_name]["status"] == "finished":
            foo[model_name]["progress"] = "100"

        elif foo[model_name]["status"] == "running":
            # This oneliner gets the last "EPOCH: N/M" occurence on the log
            # file of the training and only returns the "N/M" part
            proc = subprocess.Popen(
                f'tsp -o {model_id} | xargs grep -oE "EPOCH: [0-9]+/[0-9]+" | cut -d " " -f2 | tail -n 1',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True)
            perc_fraction = proc.stdout.readline().decode(
                'ascii').strip().split('/')
            if len(perc_fraction) == 2:
                foo[model_name]["progress"] = str(floor(100 * int(perc_fraction[0]) / int(perc_fraction[1])))
            else:
                foo[model_name]["progress"] = '0'

        else:
            foo[model_name]["status"] = "error"

        result.append(foo)

    return result

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/all-models")
async def get_all_models(user: str):
    user_models = [
        {"language": model.split("/")[1], "dataset": model.split("/")[2], "model_name": model.split("/")[3]}
        for model in get_users_models_dict(user).keys()
        if model.startswith("tts_models")
    ]
    user_models.extend(models_list)
    return user_models


@app.get("/users-models")
async def get_users_models(user: str):
    return check_user_models(user)


@app.post("/login")
async def login(username: str = Form(), password: str = Form()):
    f = open(users_config_path)
    data = json.load(f)
    response = 'OK' if username in data['users'] and data['users'][username]['password'] == password else 'ERROR'
    return {'message': response}


@app.post("/create-user")
async def create_user(username: str = Form(), password: str = Form()):
    new_user = {username: {"password": password, "models": []}}
    write_json(new_user, 'new_user')
    os.mkdir(os.path.join(users_path, username))
    return {'message': 'OK'}


@app.post("/tts")
async def get_tts_audio(
    language: str = Form(),
    dataset: str = Form(),
    model_name: str = Form(),
    text: str = Form(),
    multispeaker_lang: str = Form(),
    file: UploadFile = None,
):
    model_path = None
    config_path = None
    speakers_file_path = None
    vocoder_path = None
    vocoder_config_path = None
    speaker_wav = None

    print(" > Dataset: {}".format(dataset))
    print(" > Language: {}".format(language))
    print(" > Model name: {}".format(model_name))
    print(" > Text: {}".format(text))
    print(" > Multispeaker language: {}".format(multispeaker_lang))

    # TODO: check TTS/bin/synthesize for other options
    model_name = "tts_models/{}/{}/{}".format(language, dataset, model_name)
    model_path, config_path, model_item = manager.download_model(model_name)
    vocoder_name = model_item["default_vocoder"]
    if vocoder_name:
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

    if file:
        speaker_wav = os.path.join(
            tmp_dir, 
            datetime.now().strftime("speaker_wav_%m_%d_%Y_%H_%M_%S.wav")
        )
        # Async writing temp wav file to disk
        async with aiofiles.open(speaker_wav, 'wb') as out_file:
            content = await file.read()  # async read
            await out_file.write(content)  # async write

    # synthesize text
    wavs = synthesizer.tts(text, speaker_wav=speaker_wav, language_name=multispeaker_lang)
    output_path = os.path.join(tmp_dir, datetime.now().strftime("tts_%m_%d_%Y_%H_%M_%S.wav"))
    synthesizer.save_wav(wavs, output_path)
    return FileResponse(output_path)


@app.post("/get-database")
async def get_helena_database(name: str):
    return zip_files(name)

@app.get("/check-database-name")
async def check_database_name(username: str, db_name: str):
    response = 'ERROR' if os.path.exists(os.path.join(users_path, username, db_name)) else 'OK'
    return {'check': response}

@app.post("/train-model")
async def train_new_model(
    file: UploadFile = File(), 
    user: str = Form(), 
    language: str = Form()
):
    db_path = await unzip_files(file, user)
    model_status = train_model(db_path, language)
    return model_status
