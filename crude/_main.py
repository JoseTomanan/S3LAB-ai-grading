"""
Crude application for evaluating answers using AI. Powered by Google Gemini API.
"""
from classes import *

from dotenv import load_dotenv


IMAGE_PATH = "dataset/contour_15.jpg"
QUESTION_PATH = "dataset/2.csv"
RUBRIC_QUESTION = "Based only on the visual content of the student's work, what is the student's final answer? What is the correct mathematical answer to the problem? Are they the same?"



if __name__ == "__main__":
	load_dotenv()

	image_preprocessor = CVImagePreprocessor()
	contexter = CSVProcessor()
	ai_evaluator = AIAnswerEvaluator()

	system_prompt = ANSWER_RUBRIC_PROMPT
	rubric_question = RUBRIC_QUESTION
	image_path = IMAGE_PATH
	question_path = QUESTION_PATH

	context = contexter.get_context(question_path)
	context_question, expected_answer = context
	print("CONTEXT:", context_question, expected_answer)
	print("PROMPT:", rubric_question)
	
	image_bytes = image_preprocessor.load_image(image_path)
	image_bytes = image_preprocessor.find_first_box(image_bytes)
	image_bytes = image_preprocessor.brighten(image_bytes, amount=0.2)
	image_bytes = image_preprocessor.adjust_contrast(image_bytes, amount=1.2)

	user_prompt = f"CONTEXT:{context_question}\nPROMPT:{rubric_question}"

	item_number = ai_evaluator.get_response(image_bytes, FIND_ITEM_NUMBER_PROMPT, "")
	print("--> ITEM NUMBER:", item_number)

	response = ai_evaluator.get_response(image_bytes, system_prompt, user_prompt)
	print("--> RESPONSE:", response)
