import openpyxl
from openpyxl.styles import Border, Font, Side



class SheetsExporter:
    def __init__(self, columns: list[str], max_scores: dict[str, float | int] | None = None):
        self.columns: list[str] = columns
        self.sheet_items: dict[str, dict[str,float|None]] = {}
        self.max_scores: dict[str, float | int] | None = max_scores
        self.student_numbers: dict[str, str] = {}

    def add_student(self, key_name: str, student_no: str):
        self.sheet_items[key_name] = {i: -1 for i in self.columns}
        self.student_numbers[key_name] = student_no

    def append(self, key_name: str, num_to_score: dict[str, float|None]):
        if key_name not in self.sheet_items:
            print("WARNING: Student not in sheet; append operation aborted.")
            return
        self.sheet_items[key_name] = num_to_score

    def export_sheet(self) -> openpyxl.Workbook:
        wb = openpyxl.Workbook()

        sheet = wb.active
        assert sheet is not None
        sheet.column_dimensions['A'].width = 15
        sheet.column_dimensions['B'].width = 25

        total_col = len(self.columns) + 3

        for item in self.columns:
            sheet.cell(
                    row=1,
                    column=self.columns.index(item)+3,
                    value=item
                    )
        sheet.cell(row=1, column=total_col, value="Total")

        medium_bottom      = Border(bottom=Side(style="medium"))
        medium_right       = Border(right=Side(style="medium"))
        medium_both        = Border(bottom=Side(style="medium"), right=Side(style="medium"))
        medium_left        = Border(left=Side(style="medium"))
        medium_left_bottom = Border(left=Side(style="medium"), bottom=Side(style="medium"))

        sheet.cell(row=1, column=2).border = medium_right
        sheet.cell(row=1, column=total_col).border = medium_left

        data_start_row = 2
        if self.max_scores is not None:
            sheet.cell(row=2, column=1).border = medium_bottom
            sheet.cell(row=2, column=2).border = medium_both
            for item in self.columns:
                col = self.columns.index(item)+3
                cell = sheet.cell(row=2, column=col, value=self.max_scores.get(item, ""))
                cell.border = medium_bottom
            max_total = sum(self.max_scores.values())
            _fmt = lambda v: int(v) if isinstance(v, float) and v % 1 == 0 else v
            sheet.cell(row=2, column=total_col, value=f"{_fmt(max_total)}/{_fmt(max_total)}").border = medium_left_bottom
            data_start_row = 3

        students_list = list(self.sheet_items.keys())

        for idx, key_name in enumerate(students_list):
            row = idx+data_start_row
            sheet.cell(row=row, column=1, value=self.student_numbers[key_name])
            cell = sheet.cell(row=row, column=2, value=key_name)
            cell.border = medium_right

        for idx, key_name in enumerate(students_list):
            for item in self.columns:
                value = self.sheet_items[key_name][item]
                sheet.cell(
                        row=idx+data_start_row,
                        column=self.columns.index(item)+3,
                        value=value if value is not None else ""
                        )
            scores = self.sheet_items[key_name]
            student_total = sum(v for v in scores.values() if v is not None and v >= 0)
            if self.max_scores is not None:
                answered_items = [item for item in self.columns if scores.get(item) is not None and scores.get(item) >= 0]
                answered_max = sum(self.max_scores[item] for item in answered_items if item in self.max_scores)
                _fmt = lambda v: int(v) if isinstance(v, float) and v % 1 == 0 else v
                sheet.cell(
                        row=idx+data_start_row,
                        column=total_col,
                        value=f"{_fmt(student_total)}/{_fmt(answered_max)}"
                        ).border = medium_left

        bold = Font(bold=True)
        for col in range(1, total_col + 1):
            sheet.cell(row=1, column=col).font = bold
        total_rows = data_start_row + len(students_list)
        for row in range(1, total_rows):
            sheet.cell(row=row, column=1).font = bold

        # sheets_directory = f"./static/sheets/{file_name}.xlsx"

        # wb.save(sheets_directory)
        # print(f"--> File '{file_name}.xlsx' saved.")

        return wb
