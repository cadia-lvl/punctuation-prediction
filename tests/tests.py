from process_text import process_line

unprocessed_text = ['Snjóflóð féll nú fyrir skömmu úr Eyrarhlíð.',
					'Lokar það veginum milli Ísafjarðar og Hnífsdals?',
					'Hlynur Snorrason, yfirlögregluþjónn á Ísafirði, segir eftirfarandi!',
					'Hlynur: verið er að skoða það hvort óhætt sé að opna veginn, og þá tímabundið.']

processed_text = ['snjóflóð féll nú fyrir skömmu úr eyrarhlíð .PERIOD ', 
				  'lokar það veginum milli ísafjarðar og hnífsdals ?QUESTIONMARK ', 
				  'hlynur snorrason ,COMMA yfirlögregluþjónn á ísafirði ,COMMA segir eftirfarandi !EXCLAMATIONMARK ', 
				  'hlynur :COLON verið er að skoða það hvort óhætt sé að opna veginn ,COMMA og þá tímabundið .PERIOD ']

def test_process_line():
    assert [process_line(elem) for elem in unprocessed_text] == processed_text, "Should be equal"

if __name__ == "__main__":
    test_process_line()
    print("Everything passed")
