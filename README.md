With lambda stack:
```
git clone https://github.com/eolecvk/text-to-naruto.git
cd text-to-naruto
python3 -m pip install --upgrade pip
python3 -m pip install \
    gradio==3.5 \
    ray[serve] \
    diffusers \
    transformers \
    scipy \
    ftfy \
    datasets \
    accelerate
serve run demo:app
```


Without lambda stack:
```
git clone https://github.com/eolecvk/text-to-naruto.git
cd text-to-naruto
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install  -r requirements.txt
serve run demo:app
```
