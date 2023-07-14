fuser -n tcp -k 8519
nohup python server-fastapi.py 8519 &