import train as tr

# test_sentence = input('Masukkan kalimat : ')
# tr.predict_sentiment(test_sentence)

test_sentence1 ="Pelayanan hotel tidak ramah"
tr.predict_sentiment(test_sentence1)

test_sentence2 ="Recommended, kamar mandi, sarapannya, kamarnya lumayan."
tr.predict_sentiment(test_sentence2)

test_sentence3 ="Kamarnya nyaman"
tr.predict_sentiment(test_sentence3)