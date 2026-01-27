# import sys
# import os

# target_dir = os.path.abspath("../crude")

# if target_dir not in sys.path:
#     sys.path.insert(0, target_dir)

from model import SheetsExporter


if __name__ == "__main__":
	columns = ['1', '2a', '2b']
	sheet = SheetsExporter(columns)

	sheet.add_student("DELA CRUZ, Juan")
	sheet.add_student("RIVERO, Ricci")

	sheet.append(
			"DELA CRUZ, Juan",
			{c: 1.0 for c in columns}
		)
	
	sheet.append(
			"RIVERO, Ricci",
			{"1": 1.0, "2a": 2.25, "2b": 3.5}
		)
	
	sheet.export_sheet("TESTFILE")