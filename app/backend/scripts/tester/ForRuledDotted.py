from logic.ai_interface import AIAnswerEvaluator
from logic.box_segmenter import BoxSegmenter



FILENAMES = [
    "testRuledDottedA.jpeg",
    "testRuledDottedB.jpeg",
    "testRuledDottedC.jpeg",
    "testRuledDottedD.jpeg",
    "testRuledDottedE.jpeg",
    "testRuledDottedF.jpeg",
    ]
GET_INPUT = lambda x : f"./TEMP/input/{x}"
GET_OUTPUT = lambda x : f"./TEMP/output/{x}"



if __name__ == "__main__":
    BOX_SEGMENTER = BoxSegmenter()
    AI_EVALUATOR = AIAnswerEvaluator()
    for file in FILENAMES:
        try:
            _onlyfilename = file.split(".")[0]
            BOX_SEGMENTER.debug_dir = f"./TEMP/output/{_onlyfilename}"
            
            image_before_before = BOX_SEGMENTER.load_image(GET_INPUT(file))
            image_before = BOX_SEGMENTER.scan_page(image_before_before, debug=False)
            images_after_box = BOX_SEGMENTER.get_answer_sections(image_before, num_boxes=3, debug=True)

            for i, b in enumerate(images_after_box):
                BOX_SEGMENTER.save_image(b, GET_OUTPUT(f"{_onlyfilename}/section{i}.jpg"))
        except:
            print(f"INFO:\tFailed for {file}")
        print("================================")