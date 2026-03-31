from logic.ai_interface import AIAnswerEvaluator
from logic.box_segmenter import BoxSegmenter



FILENAMES = [
    "properTest1.jpeg",
    "properTest2.jpeg",
    ]

EXPECTED_ANSWERS: dict[str,str] = {
    "1a": "Thousands",
    "1b": "8056",
}

RUBRIC_QUESTIONS: dict[str,list[str]] = {
    "2": [
        "Correct setup of equation",
        "Correct solution",
        "Correct 'mangoes' label in final answer"
    ],
    "3": [
        "Correct setup of equation",
        "Correct solution",
        "Correct 'books' label in final answer"
    ]
}

GET_INPUT = lambda x : f"./TEMP/input/{x}"
GET_OUTPUT = lambda x : f"./TEMP/output/{x}"



if __name__ == "__main__":
    SHORT_FORM_LABELS = list(EXPECTED_ANSWERS.keys())
    PROB_SOL_LABELS = list(RUBRIC_QUESTIONS.keys())
    LABEL_CHOICES = SHORT_FORM_LABELS + PROB_SOL_LABELS
    
    BOX_SEGMENTER = BoxSegmenter()
    AI_EVALUATOR = AIAnswerEvaluator()
    
    for file in FILENAMES:
        _onlyfilename = file.split(".")[0]
        BOX_SEGMENTER.debug_dir = f"./TEMP/output/{_onlyfilename}"
        
        image_before_before = BOX_SEGMENTER.load_image(GET_INPUT(file))
        image_before = BOX_SEGMENTER.scan_page(image_before_before, debug=False)
        images_after_box = BOX_SEGMENTER.get_answer_sections(image_before, num_boxes=4, debug=True)

        for i, b in enumerate(images_after_box):
            image_beautified = BOX_SEGMENTER.beautify_scan(b)
            label = AI_EVALUATOR.get_nearest_item_number(image_beautified, label_choices=LABEL_CHOICES)
            BOX_SEGMENTER.save_image(b, GET_OUTPUT(f"{_onlyfilename}/box{i}_label{label}.jpg"))

            ## Question not given is experimental
            ## TODO: Retest with given question
            if label in SHORT_FORM_LABELS:
                res = AI_EVALUATOR.evaluate_expected_answer(image_beautified,
                                    question="(question not given for this item)",
                                    answer=EXPECTED_ANSWERS[label]
                                    )
            elif label in PROB_SOL_LABELS:
                res = AI_EVALUATOR.evaluate_multi_rubric(image_beautified,
                                    question="(question not given for this item)",
                                    rubrics=RUBRIC_QUESTIONS[label]
                                    )
            else:
                res = "NONE"
        
            print(f"BACKEND:\t{res}")
        print("================================")