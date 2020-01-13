from process_text.py import process_line

# Get the contents of a file into the correct format for processing
def read_file_string(src):
    with open(src, 'r') as file:
        lines = file.read()
    splitLines = lines.split('\n')
    return [x.split('\t') for x in splitLines]

# Now we process the input text according to the previous script
if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_string = read_file_string(sys.argv[1])
    else:
        print("There is no specified text to process")

    processed_text = [process_line(elem) for elem in file_string]

    # Write it to train, dev, and test files.
    if len(sys.argv)  == 4:
        train_text, tmp_text = train_test_split(processed_text, test_size=sys.argv[2], random_state=42)
        dev_text, test_text = train_test_split(tmp_text, test_size=sys.argv[3] random_state=42)
    else:
        train_text, tmp_text = train_test_split(processed_text, test_size=0.2, random_state=42)
        dev_text, test_text = train_test_split(tmp_text, test_size=0.5 random_state=42)

    with open("processed_text.train.txt", 'w', encoding='utf-8') as train_file:
        for item in train_text:
            train_file.write("%s\n" % item)

    with open("processed_text.dev.txt", 'w', encoding='utf-8') as dev_file:
        for item in dev_text:
            dev_file.write("%s\n" % item)

    with open("processed_text.test.txt", 'w', encoding='utf-8') as test_file:
        for item in test_text:
            test_file.write("%s\n" % item)
