Setup of the virtual einvironment:
```bash
python3.14 -m venv venv
```

Activate the virtual environment:
```bash
source ./venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Deactivate the virtual environment:
```bash
(venv) ➜  ai-python-github-copilot-sample-01 git:(main): deactivate
```

Set Hugging Face cache directory (optional, but can speed up subsequent runs):
```bash
export HF_TOKEN=[the token]
```

Run it:
```bash
(venv) ➜  ai-python-github-copilot-sample-01 git:(main): python app.py
```
