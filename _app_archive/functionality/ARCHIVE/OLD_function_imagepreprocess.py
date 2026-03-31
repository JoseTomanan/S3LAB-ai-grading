# @app.post("/api/image_preprocess")
# async def image_preprocess(file: UploadFile = File(...)):
#     """Process raw student assessment image through CV pipeline (standalone)"""
#     # Validation
#     if not file or not file.filename:
#         raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="No file provided"
#                 )
    
#     if not DOCUMENT_SCANNER.validate_file_extension(file.filename):
#         raise HTTPException(
#                 status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
#                 detail=f"Unsupported file format. Allowed: .jpg, .jpeg, .png. Got: {file.filename}"
#                 )
    
#     try:
#         contents = await file.read()
#         if len(contents) == 0:
#             raise HTTPException(
#                     status_code=status.HTTP_400_BAD_REQUEST,
#                     detail="Uploaded file is empty"
#                     )
#     except Exception as e:
#         raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail=f"Failed to read file: {str(e)}"
#                 )
    
#     # Processing
#     try:
#         preprocessor = CVImagePreprocessor(
#             use_paddle_ocr=USE_PADDLE_OCR,
#             paddle_ocr_lang=PADDLE_OCR_LANG,
#             debug_mode=False  # Set to True for debugging during development
#         )
#         processed_list = preprocessor.process_assessment_image(contents)
#     except CVProcessingError as e:
#         raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail=f"CV processing failed: {str(e)}"
#                 )
#     except Exception as e:
#         raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail=f"Unexpected error during image processing: {str(e)}"
#                 )
    
#     # Save to temp files
#     session_id = str(uuid.uuid4())
#     boxes_info = []
    
#     for i, img_bytes in enumerate(processed_list):
#         filename = f"{session_id}_{i}.jpg"
#         filepath = TEMP_DIR / filename
        
#         with open(filepath, "wb") as f:
#             f.write(img_bytes)
        
#         boxes_info.append({
#                 "index": i,
#                 "image_directory": f"/api/temp/{filename}"
#                 })
    
#     return {
#             "num_boxes": len(processed_list),
#             "boxes": boxes_info
#             }