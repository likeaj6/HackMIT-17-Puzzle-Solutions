import base64
import json

with open('captcha.json') as data_file:
    json_file = json.load(data_file)

    # i = 0

    for pic in json_file['images']:
        name = pic['name']
        fh = open(name+ ".png", "wb")
        fh.write(base64.decodestring(pic['jpg_base64']))
        fh.close()
        # i+= 1
