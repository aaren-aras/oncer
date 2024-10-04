
Make sure different Python versions installed are set up in the PATH environment variable.

No luck with using `venv` to create a virtual environment for Python 3.8.10, so `virtualenv` is used instead.
```bash
$ cd api && virtualenv -p .../Python/Python38/python.exe .venv
$ .venv/Scripts/activate  # forward slash for unix-like (Git Bash), backslash for windows (cmd)
$ pip install -r requirements.txt
```

To generate model files, run the following command:
```bash
$ cd api/src/services && python prepare_model.py
```