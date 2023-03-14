import train as tr

# test_sentence = input('Masukkan kalimat : ')
# tr.predict_sentiment(test_sentence)

# test_sentence1 ="Pelayanan hotel tidak ramah"
# s = tr.predict_sentiment(test_sentence1)
# print(s)

test_sentence2 ="Recommended, kamar mandi, sarapannya, kamarnya lumayan."
s = tr.predict_sentiment(test_sentence2)
print(s)

test_sentence3 ="Kamarnya nyaman"
tr.predict_sentiment(test_sentence3)