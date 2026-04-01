import openpyxl
from openpyxl.styles import Border, Side



class SheetsExporter:
    def __init__(self, columns: list[str], max_scores: dict[str, float | int] | None = None):
        self.columns: list[str] = columns
        self.sheet_items: dict[str, dict[str,float|None]] = {}
        self.max_scores: dict[str, float | int] | None = max_scores

    def add_student(self, key_name: str):
        self.sheet_items[key_name] = {i: -1 for i in self.columns}

    def append(self, key_name: str, num_to_score: dict[str, float|None]):
        if key_name not in self.sheet_items:
            print("WARNING: Student not in sheet; append operation aborted.")
            return
        self.sheet_items[key_name] = num_to_score

    def export_sheet(self) -> openpyxl.Workbook:
        wb = openpyxl.Workbook()

        sheet = wb.active
        assert sheet is not None
        sheet.column_dimensions['A'].width = 25

        for item in self.columns:
            sheet.cell(
                    row=1,
                    column=self.columns.index(item)+2,
                    value=item
                    )

        medium_bottom = Border(bottom=Side(style="medium"))
        medium_right  = Border(right=Side(style="medium"))
        medium_both   = Border(bottom=Side(style="medium"), right=Side(style="medium"))

        data_start_row = 2
        if self.max_scores is not None:
            for item in self.columns:
                col = self.columns.index(item)+2
                cell = sheet.cell(row=2, column=col, value=self.max_scores.get(item, ""))
                cell.border = medium_bottom
            sheet.cell(row=2, column=1).border = medium_both
            data_start_row = 3

        students_list = list(self.sheet_items.keys())

        for idx, key_name in enumerate(students_list):
            cell = sheet.cell(row=idx+data_start_row, column=1, value=key_name)
            cell.border = medium_right

        for idx, key_name in enumerate(students_list):
            for item in self.columns:
                value = self.sheet_items[key_name][item]
                sheet.cell(
                        row=idx+data_start_row,
                        column=self.columns.index(item)+2,
                        value=value if value is not None else ""
                        )

        # sheets_directory = f"./static/sheets/{file_name}.xlsx"

        # wb.save(sheets_directory)
        # print(f"--> File '{file_name}.xlsx' saved.")

        return wb