from process_text import process_line
from sklearn.model_selection import train_test_split
import sys

# We process the input text according to the processing script

try:
    if len(sys.argv) > 1:
        float(sys.argv[1])
        print("There is no specified text to process")
        sys.exit(0)
except ValueError:
    with open(sys.argv[1], 'r') as file:
        lines = file.read()
    file_string =  lines.split('\n')

if not isinstance(file_string[0], str):
    print("The input text needs to be a list of strings")
    sys.exit(0)

print(f'Number of rows in file: {len(file_string)}')

# Punctuation marks processed
processed_text = [process_line(elem) for elem in file_string]

# Save processed_text to file
with open("workdir/processed_text.all.txt", 'w', encoding='utf-8') as all_file:
    for item in processed_text:
        all_file.write("%s\n" % item)

print("Done processing the text")

# Write it to train, dev, and test files.
if len(sys.argv)  > 3:
    train_text, tmp_text = train_test_split(processed_text, test_size=float(sys.argv[2]), random_state=42)
    dev_text, test_text = train_test_split(tmp_text, test_size=float(sys.argv[3]), random_state=42)
else:
    train_text, tmp_text = train_test_split(processed_text, test_size=0.2, random_state=42)
    dev_text, test_text = train_test_split(tmp_text, test_size=0.5, random_state=42)

with open("datadir/processed_text.train.txt", 'w', encoding='utf-8') as train_file:
    for item in train_text:
        train_file.write("%s\n" % item)

with open("datadir/processed_text.dev.txt", 'w', encoding='utf-8') as dev_file:
    for item in dev_text:
        dev_file.write("%s\n" % item)

with open("datadir/processed_text.test.txt", 'w', encoding='utf-8') as test_file:
    for item in test_text:
        test_file.write("%s\n" % item)

print("Done saving files to data directory")
