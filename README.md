# AI grading

A wrapper for Google Gemini API that assesses a student's answer, given a prompt. The prompt could either check for the final answer of the student (and compare it with an expected answer/its own answer), or to see if the student was able to reach a particular step in the process.

## How to use
To use the Gemini API, you would need to obtain a key and place it in a `.env` file with variable name `GEMINI_API_KEY`. The file should now contain a line that looks like the following:

```txt
GEMINI_API_KEY=<your key here>
```
Then, run the following commands in shell.

### Install Python dependencies
```shell
pip install -r requirements.txt
```

### Run application
```shell
python ./crude/main.py
```