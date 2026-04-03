### **Main user flow**

1. Register section
2. Add students for each section
3. Create test instance
4. Fill up test item details:
    1. Item type
    2. Label
    3. Expected answer / rubric questions
5. Outside of the application…
    1. Take photos
    2. \[N2H\] Rename according to student number
    3. \[N2H\] If necessary, rotate orientation
6. Bulk upload photos from gallery
    1. If need, rename according to student number
    2. If need, rotate orientation/flip image
7. Send for segmentation and AI evaluation
8. Verify results
    1. If need, recapture/redo/manual crop
9. Check scores
10. If wrong assessment, edit manually — this should have an indicator
11. Download spreadsheet

### **What currently works**

* *Background:*
    * Dark/deeply contrasting with pad paper
    * No light-colored blobs
* *Papers used:*
    * Grade 3 pad paper or plain white paper
    * Photo orientation aligned with answer sheet (i.e., if portrait answer sheet, then portrait taking of photo)
    * Minimal occurence of curling & dog ears
* *Answers:*
    * Marked with solid dots in corners
    * Dots are drawn with marker/pen
    * Dots are 4x thickness of writings/rules of pad paper
* *Answer segments:*
    * Box-shaped
    * 1 per answer, non-split

### **What may not work**

* *Background:*
    * Blends with contour of pad paper
    * Has a bigger box around the picture (this will get detected instead of the actual image)
* *Papers used:*
    * Crumpled
    * Inaccurate orientation (e.g., accidentally portrait)
    * Excessive curling, dog ears
* *Answers:*
    * Dots are drawn with pencil
    * Dot thickness is \< 4x of writings/rules of pad paper
* *Answer segments:*
    * Non-rectangular
    * Split into multiple parts