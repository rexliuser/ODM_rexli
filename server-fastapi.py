from typing import List
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import os, subprocess, sys
from cryptography.fernet import Fernet

path_script = os.path.dirname(os.path.realpath(__file__))
path_odm = os.path.dirname(path_script)
path_project = os.path.join(path_odm,'projects')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

subprocess_proc = None

def session_to_project_name(session):
    return f'webproject_{session}_aws'

@app.post("/createProject")
async def upload(files: List[UploadFile] = File(...), session: str = Form(...), unmask: str = Form(...), ):
    projectname = session_to_project_name(session)
    path_projectname = os.path.join(path_project,projectname)
    
    code = Fernet.generate_key().decode('utf-8')[:12]  
    
    if unmask!='false':
        commandlist = ['python','run.py',projectname,'-s',code,'--aws']
    else:
        commandlist = ['python','run.py',projectname,'-s',code,'-m','--aws']
    
    if not os.path.exists(path_projectname):
        os.mkdir(path_projectname)
        path_dataset = os.path.join(path_projectname,'dataset')
        os.mkdir(path_dataset)
        
        for file in files:
            with open(os.path.join(path_dataset,file.filename),'wb') as f:
                f.write(await file.read())
        
        try:
            global subprocess_proc
            subprocess_proc = subprocess.Popen(commandlist)
        except subprocess.CalledProcessError as e:
            app.logger.error(str(e))
            return JSONResponse(content={'error': 'An error occured.'},status_code=500)
        return JSONResponse(content={'code': code},status_code=200)


@app.get('/files/{filename}')
async def load_model(filename: str):
    filepath = os.path.join('../sample_models/',filename)
    headers = {"Content-Disposition": "attachment; filename="+filename}
    if os.path.exists(filepath):
        return FileResponse(filepath, headers=headers)
    else:
        return JSONResponse(content={'message':'404 File Not Found'}, status_code=404)
    
@app.get('/download/{code}')
async def download_file(code: str):
    filepath = os.path.join('../output_models/',code,'mesh.glb')
    headers = {"Content-Disposition": "attachment; filename=mesh.glb"}
    if os.path.exists(filepath):
        return FileResponse(filepath, headers=headers)
    else:
        return JSONResponse(content={'message':'404 File Not Found'}, status_code=404)

@app.get('/testserver')
async def server_running():
    return JSONResponse(content={"message":"Server Running."}, status_code=200)

@app.get('/progress/{session}')
async def progress(session: str):
    if session_to_project_name(session)[-3:]=='aws':
        output_path = os.path.join(path_odm,'output_models',session,'mesh.glb')
        if os.path.exists(output_path):
            return_progress = str(1)
        else:
            return_progress = str(0)
    else:
        path = '../projects/webproject_'+session
        max_no=800
        if os.path.exists(os.path.join(path,'exports','mesh.glb')):
            count = max_no
        else:
            count = min(776,count_files(session))
        return_progress = str(count/max_no)
    return JSONResponse(content={"message":return_progress}, status_code=200)

def count_files(session: str):
    path = '../projects/webproject_'+session
    count = 0
    for root, dirs, files in os.walk(path):
        count += len(files)
    return count

def get_private_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8',80))
    ip = s.getsockname()[0]
    s.close()
    return ip

if __name__ == "__main__":
    import uvicorn 
    host = get_private_ip()
    port = str(sys.argv[-1]) if len(str(sys.argv[-1]))==4 else '8519'
    uvicorn.run(app, host=host, port=int(port))
    