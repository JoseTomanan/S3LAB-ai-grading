# AI grading

A wrapper for Google Gemini API that assesses a student's answer, given a prompt (that could either check for the final answer of the student and see if it's correct, or to see if some of the steps followed were correct).

## How to use
To use the Gemini API, you would need to obtain a key and place it in a `.env` file in the root directory as follows:

```txt
GEMINI_API_KEY=<your key here>
```
Then, run the following commands in shell.

## Install Python dependencies
```sh
pip install -r requirements.txt
```

## Run application
```sh
python ./crude/main.py
```