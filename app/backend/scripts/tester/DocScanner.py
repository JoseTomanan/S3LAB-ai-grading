from logic.document_scanner import DocumentScanner

FILENAMES = [
    "202090011.jpeg",
    "202090022.jpeg",
    "properTest1.jpeg",
    "properTest2.jpeg",
    "badScan1.jpg",
    "goodScan1.jpeg",
    "goodScan2.jpeg",
    "goodScan3.jpeg",
    "goodScan4.jpg",
]
GET_INPUT = lambda x : f"./TEMP/input/{x}"
GET_OUTPUT = lambda x : f"./TEMP/output/{x}"



if __name__ == "__main__":
    DOCUMENT_SCANNER = DocumentScanner()

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
        if file == FILENAMES[-1]:
            print("--------------------------------")
    print("================================")