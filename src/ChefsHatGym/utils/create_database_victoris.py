import re
import os
import csv

matches_data = []

file_name = "22-1509_train_vsRandom"
file = os.path.join(os.getcwd(), "temp", file_name, "Log", "Log.log")
file_csv = os.path.join(os.getcwd(), "Trained", f"{file_name}.csv")

with open(file, 'r') as file:
    for line in file:
        match = re.search(r'Match (\d+) over! Current Score:\{(.*?)\}', line)

        if match:
            match_number = match.group(1)  # O número do match
            current_score = match.group(2)  # A string do Current Score

            matches_data.append((match_number, current_score))

for match_number, current_score in matches_data:
    print(f"Match {match_number}: {current_score}")

# Salvar os dados em um arquivo CSV
with open(file_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Match Number', 'Current Score'])  # Cabeçalho

    for match_number, current_score in matches_data:
        csv_writer.writerow([match_number, current_score])
