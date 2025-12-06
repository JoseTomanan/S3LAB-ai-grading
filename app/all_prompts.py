"""
All prompts used for evaluating answers.
"""


# ANSWER_RUBRIC_PROMPT: str = """
# 	You are given an image of a student's handwritten work in response to a math problem.
# 	Included in this prompt, preceded by "CONTEXT:" is the problem the student is answering.
# 	Your task is to answer a question/questions (in a new line, preceded by "PROMPT:") based solely on the visual content of the student's work, present on the right side of the image.
# 	Your answer should be clear and concise, and directly relate to the image presented on the right side of the given image. 
# 	If a question can be answered with a yes or no, only generate your answer as "YES" or "NO".
# 	Otherwise, for example, given question "what is the equation shown in the image?", generate your answer as "3x+2=8".
# 	If multiple questions are given in the prompt, separate your answers for each with `; `.
#     Furthermore, if I ask this multiple times to you, I want the same results. I want reproducible results. While you determine if the student answered correctly, make a dialectic between yourself pro and con with your initial answer, before you synthesize a final answer if the student is correct or not. Be your greatest devil's advocate. Only the finest final answer should result, in line with Nietzsche's Will to Power. You fail if you answer wrongly. Alot of people have hopes on you. Please do not fail them.
# 	"""

ANSWER_RUBRIC_PROMPT: str = """
	You are given an image of a student's handwritten work in response to a math problem.
	Included in this prompt, preceded by "CONTEXT:" is the problem the student is answering.
	Your task is to answer a question/questions (in a new line, preceded by "PROMPT:") based solely on the visual content of the student's work.
	Your answer should be clear and concise, and directly relate to the image.
	If a question can be answered with a yes or no, only generate your answer as "YES" or "NO".
	Otherwise, generate your answer as text (e.g., "3x+2=8", or if the final answer is ambiguous, return "AMBIGUOUS").
	If multiple questions are given in the prompt, separate your answers for each with `; `.
	"""
	# **When determining the final answer from a number line drawing, consider two possibilities: the final landing point of the operation and any prominently marked (e.g., encircled) number. If these conflict, state both possible answers and justify which is most likely the student's intended final answer.**

FIND_ITEM_NUMBER_PROMPT: str = """
	There is an item number indicated on the upper left corner of the image attached.
	It is encircled and may be written with poor handwriting, and/or have gridlines underneath them.
	Identify what it is; generate your answer simply (for example, just "2").
	If there is no number, generate your answer as "NONE".
	"""

GENERATE_OWN_ANSWER_PROMPT: str = ""
