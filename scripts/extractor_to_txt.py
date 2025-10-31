import os
import csv

def extract_csv_column_to_txt(csv_path: str, column_name: str, output_path: str) -> None:
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	# Ensure output directory exists
	os.makedirs(os.path.dirname(output_path), exist_ok=True)

	extracted_entries = []

	# Read the CSV file
	with open(csv_path, mode="r", encoding="utf-8", newline="") as csvfile:
		reader = csv.DictReader(csvfile)
		if column_name not in reader.fieldnames: # pyright: ignore[reportOperatorIssue]
			raise ValueError(f"Column '{column_name}' not found in CSV headers: {reader.fieldnames}")
	
		for row in reader:
			value = row[column_name].strip() if row[column_name] is not None else ""
			extracted_entries.append(value)
		
	# Write entries to .txt file (one per line)
	with open(output_path, mode="w", encoding="utf-8") as txtfile:
		for entry in extracted_entries:
			txtfile.write(entry + "\n")

	print(f"Extracted column {column_name} from '{csv_path}' saved to '{output_path}'")