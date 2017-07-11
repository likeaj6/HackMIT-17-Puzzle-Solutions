import requests
import string

trueMessage = "the docker of ecommerce but like duolingo"
key = "is a missing can major worked four nuttcase after front"
currentIndex = 11
words = []

with open('transcript.txt', 'r') as myfile:
    data = myfile.read().replace('\n', ' ')
    data = data.translate(None, string.punctuation)
    currentMessage = trueMessage[0:currentIndex]
    for word in data.split(" "):
        if word.lower() not in words:
            # print word
            words.append(word.lower())
            attempt = key + " " + word
            print(attempt)
            result = requests.post('https://the.delorean.codes/api/decode', data={"username":"likeaj6", "codeword": attempt}).json()
            if result['well_formed'] == True:
                print result['message']
                if result['message'] == trueMessage[0:currentIndex]:
                    print("SUCCESS!!!!!")
                    print result['message']
                    key += " " + word
                    currentIndex += 1
            if result['message'] == trueMessage:
                print(attempt)
                break
