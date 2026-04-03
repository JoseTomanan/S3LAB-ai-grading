from pathlib import Path
from logic.box_segmenter import BoxSegmenter



GET_INPUT = lambda x : f"./TEMP/input/{x}"
GET_OUTPUT = lambda x : f"./TEMP/output/{x}"
FILENAMES = [
    f.name for f in Path("./TEMP/input").iterdir()
    if f.name.lower().startswith("proper")
    and f.name.lower().endswith(('.jpeg', '.jpg', '.png'))
]



if __name__ == "__main__":
    BOX_SEGMENTER = BoxSegmenter()

    for file in FILENAMES:
        _onlyfilename = file.split(".")[0]
        BOX_SEGMENTER.debug_dir = f"./TEMP/output/{_onlyfilename}"
        
        image_before_before = BOX_SEGMENTER.load_image(GET_INPUT(file))
        image_before = BOX_SEGMENTER.scan_page(image_before_before, debug=False)
        images_after_box = BOX_SEGMENTER.get_answer_sections(image_before, num_boxes=4, debug=True)

        for i, b in enumerate(images_after_box):
            image_beautified = BOX_SEGMENTER.beautify_scan(b)
            BOX_SEGMENTER.save_image(b, GET_OUTPUT(f"{_onlyfilename}/boxed/{i}.jpg"))
        print("================================")