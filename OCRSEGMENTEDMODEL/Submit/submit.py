import requests
import json

url = 'https://captcha.delorean.codes/u/likeaj6/solution'

with open('new.json') as data_file:
    json_file = json.load(data_file)
    headers = {'content-type': 'application/json'}
    response = requests.post(url, data=json.dumps(json_file), headers=headers)
    print(response)
    print(response.content)
