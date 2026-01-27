"""
Crude application for evaluating answers using AI. Powered by Google Gemini API.
"""
from model import *
from dotenv import load_dotenv


BASE_PATH = "./crude/input"
SYSTEM_PROMPT = ANSWER_RUBRIC_PROMPT
IMAGE_PATH = f"{BASE_PATH}/test_good.jpg"
QUESTION_PATH = f"{BASE_PATH}/item.csv"
RUBRIC_QUESTION = "Based only on the visual content of the student's work, what is the student's final answer? What is the correct mathematical answer to the problem? Are they the same?"



if __name__ == "__main__":
	load_dotenv()

	context = CSVProcessor().get_context(QUESTION_PATH)
	user_prompt = f"QUESTION: {context[0]}\nPROMPT: {RUBRIC_QUESTION}"

	print("--> USER PROMPT:", user_prompt)
	
	image_bytes = CVImagePreprocessor().load_image(IMAGE_PATH)
	image_bytes = CVImagePreprocessor().find_first_box(image_bytes)
	image_bytes = CVImagePreprocessor().brighten(image_bytes, amount=0.2)
	image_bytes = CVImagePreprocessor().adjust_contrast(image_bytes, amount=1.2)

	item_number = AIAnswerEvaluator().get_response(image_bytes, FIND_ITEM_NUMBER_PROMPT, "")
	print("--> ITEM NUMBER:", item_number)

	response = AIAnswerEvaluator().get_response(image_bytes, SYSTEM_PROMPT, user_prompt)
	print("--> RESPONSE:", response)
