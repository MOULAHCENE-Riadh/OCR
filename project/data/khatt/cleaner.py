import csv

input_path = 'project/data/khatt/labels.csv'
output_path = 'project/data/khatt/labels_clean.csv'

with open(input_path, 'r', encoding='utf-8') as infile, \
     open(output_path, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
    header = next(reader)
    writer.writerow(header)
    for row in reader:
        # Merge all columns after the first into the text field
        if len(row) > 2:
            row = [row[0], ','.join(row[1:])]
        writer.writerow(row)