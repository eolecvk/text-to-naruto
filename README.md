Setup:
```
virtualenv -p /usr/bin/python3.8 .venv
. .venv/bin/activate
python3 pip install --upgrade pip
python3 pip install  -r requirements.txt
```

Deploy:
```
serve run demo:app
```