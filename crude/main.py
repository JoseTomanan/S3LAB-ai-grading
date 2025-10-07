"""
Property of Tomanan, Tuan at S3Labs of University of Philippines - Diliman.
"""
import sys

sys.path.append("..")

from openai import OpenAI

class OpenAIImageToText:
	...

if __name__ == "__main__":
	path = ...

	prompt = """
			You are a teacher and are given a student's handwritten work in an image format. The handwritten work is a response 
			to a problem. Write a description for this image to explain everything in the image of the student's handwritten work 
			in as much detail as you can so that another teacher can understand and reconstruct the math work in this image without 
			viewing the image. Focus on describing the student's answers in the image. Your response should be a paragraph without bullet points.
		"""
	
	openai_api = OpenAIImageToText()
	response = openai_api.generate(prompt=prompt.format(path=path))