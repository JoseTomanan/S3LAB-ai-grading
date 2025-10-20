"""
All prompts used for evaluating answers.
"""


ANSWER_RUBRIC_PROMPT: str = """
	You are given an image of a student's handwritten work in response to a math problem.
	Included in this prompt, preceded by "CONTEXT:" is the problem the student is answering.
	Your task is to answer a question/questions (in a new line, preceded by "PROMPT:") based solely on the visual content of the student's work, present on the right side of the image.
	Your answer should be clear and concise, and directly relate to the image presented on the right side of the given image. 
	If a question can be answered with a yes or no, only generate your answer as "YES" or "NO".
	Otherwise, for example, given question "what is the equation shown in the image?", generate your answer as "3x+2=8".
	If multiple questions are given in the prompt, separate your answers for each with `; `.
	"""

FIND_ITEM_NUMBER_PROMPT: str = """
	There is an item number indicated on the upper left corner of the image attached.
	It is encircled and may be written with poor handwriting, and/or have gridlines underneath them.
	Identify what it is; generate your answer simply (for example, just "2").
	If there is no number, generate your answer as "NONE".
	"""

GENERATE_OWN_ANSWER_PROMPT: str = ""
