import os
from logic.document_scanner import DocumentScanner

GET_INPUT = lambda x : f"./TEMP/input/{x}"
GET_OUTPUT = lambda x : f"./TEMP/output/{x}"
FILENAMES = [f for f in os.listdir(GET_INPUT("")) if os.path.isfile(os.path.join(GET_INPUT(""), f))]



if __name__ == "__main__":
    DOCUMENT_SCANNER = DocumentScanner()

    print("Detected files:", FILENAMES)
    print("================================")
    for file in FILENAMES:
        try:
            _onlyfilename = file.split(".")[0]
            DOCUMENT_SCANNER.debug_dir = f"./TEMP/output/{_onlyfilename}"
            
            image_before_before = DOCUMENT_SCANNER.load_image(GET_INPUT(file))
            image_before = DOCUMENT_SCANNER.scan_page(image_before_before, debug=True)
            DOCUMENT_SCANNER.save_image(image_before, GET_OUTPUT(f"DEBUG/SCANNER/{_onlyfilename}.jpg"))
        except:
            print(f"DEBUGGER:\t--> failed for {file}")
        print("--------------------------------")
    print("================================")