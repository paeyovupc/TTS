import os
import io
import json
import tempfile
from datetime import datetime
import zipfile
import aiofiles

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

# update in-use models to the specified released models.
model_path = None
config_path = None
speakers_file_path = None
vocoder_path = None
vocoder_config_path = None

users_config_path = os.path.dirname(__file__).replace('.', 'users_config.json')
users_path = os.path.dirname(__file__).split('api')[0].replace("TTS", "users")
databases_path = os.path.dirname(__file__).split('api')[0].replace("TTS", "databases")
models_config_path = os.path.join(os.path.dirname(__file__).replace("api", "TTS"), ".models.json")
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
        data["users"][user]["models"] = new_data

    jsonFile = open(users_config_path, "w+")
    jsonFile.write(json.dumps(data, indent=2))
    jsonFile.close()

def zip_files(db_name: str):
    db_path = os.path.join(databases_path, db_name)
    zip_filename = "{}.zip".format(db_name)
    s = io.BytesIO()
    zf = zipfile.ZipFile(s, "w")

    #Create folder inside zip
    zf.write(db_path, db_name)

    # Iterate over all the files in directory
    for folderName, subfolders, filenames in os.walk(db_path):
        for filename in filenames:
           filePath = os.path.join(folderName, filename)
           zf.write(filePath, os.path.join(db_name, filename))
    zf.close()

    # Grab ZIP file from in-memory, make response with correct MIME-type
    resp = Response(s.getvalue(), media_type="application/zip", headers={
        'Content-Disposition': f'attachment;filename={zip_filename}'
    })

    return resp

async def unzip_files(file: UploadFile, user_name: str):
    db_name = file.filename.split('.')[0]
    zip_path = os.path.join(users_path, user_name, file.filename)
    db_path = os.path.join(users_path, user_name, db_name)

    # Async writing zip file to disk 
    async with aiofiles.open(zip_path, 'wb') as out_file:
        content = await file.read()  # async read
        await out_file.write(content)  # async write

    # Unzip files and delete zip
    zip = zipfile.ZipFile(zip_path)
    zip.extractall(db_path)
    zip.close()
    os.remove(zip_path)
    
    # Create symlink /users/user_name/db_name <--> /databases/db_name
    os.symlink(db_path, os.path.join(databases_path, db_name))

@app.get("/")
async def root():
    return {"message": "Hello World"}

# TODO: add personalized models
@app.get("/all_models")
async def get_all_models():
    return models_list

@app.get("/users-models")
async def get_users_models(user: str):
    f = open(users_config_path)
    data = json.load(f)
    return data['users'][user]['models']

@app.post("/login")
async def login(username: str = Form(), password: str = Form()):
    f = open(users_config_path)
    data = json.load(f)
    response = 'OK' if username in data['users'] and data['users'][username]['password'] == password else 'ERROR'
    return {'message': response}

@app.post("/create-user")
async def create_user(username: str = Form(), password: str = Form()):
    new_user = {
        username: {
            "password": password,
            "models": []
        }
    }
    write_json(new_user, 'new_user')
    os.mkdir(os.path.join(users_path, username))
    return {'message': 'OK'}


@app.post("/tts")
async def get_tts_audio(
    language: str = Form(),
    dataset: str = Form(),
    model_name: str = Form(),
    text: str = Form(),
    file: UploadFile | None = None
    ):
    print(" > Dataset: {}".format(dataset))
    print(" > Language: {}".format(language))
    print(" > Model name: {}".format(model_name))
    print(" > Text: {}".format(text))

    # TODO: check TTS/bin/synthesize for other options
    model_name = "tts_models/{}/{}/{}".format(language, dataset, model_name)
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

    # synthesize text
    wavs = synthesizer.tts(text)
    tmp_dir = tempfile._get_default_tempdir()
    output_path = os.path.join(tmp_dir, datetime.now().strftime("tts_%m_%d_%Y_%H_%M_%S.wav"))
    synthesizer.save_wav(wavs, output_path)
    return FileResponse(output_path)  

@app.post("/get-database")
async def get_helena_database(name: str):
    return zip_files(name)
    
@app.post("/new-db-multispeaker")
async def new_db_multispeaker(file: UploadFile = File(), user: str = Form()):
    await unzip_files(file, user)
    return {'message': 'Database created!'}
    