import requests
import sys
import time
import pandas as pd
import numpy as np
import datetime

word_url = 'http://index.baidu.com/api/SearchApi/thumbnail?area=0&word={}'
# word_url1 = f'http://index.baidu.com/api/SearchApi/thumbnail?area=0&word=[[%7B%22name%22:%22{}%22,%22wordType%22:1%7D]]'
COOKIES = 'BIDUPSID=46B9A722417469754DAD0C145837E88E; PSTM=1591711542; BAIDUID=46B9A72241746975F32F72C3D9D40DFC:FG=1; BAIDUID_BFESS=46B9A72241746975F32F72C3D9D40DFC:FG=1; __yjs_duid=1_32f3b00b169b1bd0cc672687c8825c4c1619580696833; Hm_lvt_d101ea4d2a5c67dab98251f0b5de24dc=1620805967; BDUSS=zhVTTVXcmN5NH5kNHZMfkVxWC03YTZnM3FZTmpvZUQ0cDc4T1c4bklacWVGc05nSUFBQUFBJCQAAAAAAAAAAAEAAAAWG6mgt8-7sHbKwrK7tuAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJ6Jm2CeiZtgS; BDUSS_BFESS=zhVTTVXcmN5NH5kNHZMfkVxWC03YTZnM3FZTmpvZUQ0cDc4T1c4bklacWVGc05nSUFBQUFBJCQAAAAAAAAAAAEAAAAWG6mgt8-7sHbKwrK7tuAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJ6Jm2CeiZtgS; CHKFORREG=dce6648e99e8c0e150908694b1d254e6; bdindexid=lece9o3jsko83v5amf63o818m5; Hm_lpvt_d101ea4d2a5c67dab98251f0b5de24dc=1620806100; ab_sr=1.0.0_Zjg0ZDc5NjBhNDhkYWE4ODFlZjk2OGMxOTI0MDNkNWE3MGVhNjg4OWVmYzlmZWNkMTA0MGNlMDcwYjEzYmRkNmE5YWM3YjVjZWQ3YjllYTk2ZDg2M2JkNzJjZDRkMTZj; __yjs_st=2_NDBiM2IzNTBhZDUwMjAzYjZmZjY2MjFkZDAyYzU0NzZlNWQ4OWFmMGMwNzdlZWFmZTRiMmNjOTQzNTdmYmRkMTNlNzk1MGNmZTVlYzZlOGU4N2Y2ZjJiYjJiMGU5YjM4NDEzMzVlMmVlYWIwNjM0MTU3OGM0MjFhYWU5MTAxNzcyNzkyMTAyYTFlNWZiYzNlOGJmZTUyYmE4OTkwZTRhZTg0MzBmZTkyZjBhOWNkMGY4MDkwZjM1ZGRkNzYyZTk4OTA3NzA1MDU2YmQyODdmNTNiYjBhNmUxZWFkZWNmZTMyNTEzYjkwYjVmM2JmZWEyZTNhY2EzOGYxMzVlMzYwOV83X2E2MDAxMmEz; RT="z=1&dm=baidu.com&si=2yrqbkmjajj&ss=kol60w0h&sl=p&tt=m42&bcn=https%3A%2F%2Ffclog.baidu.com%2Flog%2Fweirwood%3Ftype%3Dperf"'

    # 'BIDUPSID=6C34DA33F329ACF74270250DDA77C712; PSTM=1589523676; BAIDUID=BB5A781560A929325CCF14881D661AB4:FG=1; BDUSS=9GWlVwV2UyaFRqWjhqYlVOU1ZWSnNEOFBEbFpqbkg1fkoxOFlMY2FOcUFPT2RlRVFBQUFBJCQAAAAAAAAAAAEAAADUnXqaY2doaGhjZ2hoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAICrv16Aq79eV0; BDORZ=FFFB88E999055A3F8A630C64834BD6D0; H_PS_PSSID=; Hm_lvt_d101ea4d2a5c67dab98251f0b5de24dc=1591187994,1591610816; bdindexid=rfj4nvkpb8il9sl3ii6sm40tv2; Hm_lpvt_d101ea4d2a5c67dab98251f0b5de24dc=1591610824; delPer=0; PSINO=2; BDRCVFR[1kRcOFa5hin]=mk3SLVN4HKm; RT="sl=0&ss=kb6da8yx&tt=0&bcn=https%3A%2F%2Ffclog.baidu.com%2Flog%2Fweirwood%3Ftype%3Dperf&z=1&dm=baidu.com&si=9bzana5mhzs&ld=9qm&ul=9xgx"'


def decrypt(t, e):
    n = list(t)
    i = list(e)
    a = {}
    result = []
    ln = int(len(n) / 2)
    start = n[ln:]
    end = n[:ln]
    for j, k in zip(start, end):
        a.update({k: j})
    for j in e:
        result.append(a.get(j))
    return ''.join(result)


def get_index_home(keyword):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.90 Safari/537.36',
        'Cookie': COOKIES
    }

    word_url = f'http://index.baidu.com/api/SearchApi/thumbnail?area=0&word=[[%7B%22name%22:%22{keyword}%22,%22wordType%22:1%7D]]'
    resp = requests.get(word_url, headers=headers)
    j = resp.json()

    print(j)

    uniqid = j.get('data').get('uniqid')
    return get_ptbk(uniqid)


def get_ptbk(uniqid):
    url = 'http://index.baidu.com/Interface/ptbk?uniqid={}'
    ptbk_headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cache-Control': 'no-cache',
        'Cookie': COOKIES,
        'DNT': '1',
        'Host': 'index.baidu.com',
        'Pragma': 'no-cache',
        'Proxy-Connection': 'keep-alive',
        'Referer': 'http://index.baidu.com/v2/index.html',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.90 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
    }
    resp = requests.get(url.format(uniqid), headers=ptbk_headers)
    if resp.status_code != 200:
        print('获取uniqid失败')
        sys.exit(1)
    return resp.json().get('data')


def get_index_data(keyword, start='2020-01-01', end='2020-05-01'):
    url = f'http://index.baidu.com/api/SugApi/sug?inputword[]={keyword}&area=0&startDate={start}&endDate={end}'
    word_param = f'[[%7B"name":"{keyword}","wordType":1%7D]]'
    url1 = f'http://index.baidu.com/api/SearchApi/index?area=0&word={word_param}&startDate={start}&endDate={end}'
    print(url1 + "\n")
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cache-Control': 'no-cache',
        'Cookie': COOKIES,
        'DNT': '1',
        'Host': 'index.baidu.com',
        'Pragma': 'no-cache',
        'Proxy-Connection': 'keep-alive',
        'Referer': 'http://index.baidu.com/v2/index.html',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
    }

    resp = requests.get(url1, headers=headers)
    if resp.status_code != 200:
        print('获取指数失败')
        sys.exit(1)

    # print('\n======= json ======\n')
    # print(resp.json())

    data = resp.json().get('data').get('userIndexes')[0]
    uniqid = resp.json().get('data').get('uniqid')

    # print('\n======= data ======\n')
    # print(data)
    #
    # print('\n======= uniqid ======\n')
    # print(uniqid)


    ptbk = get_ptbk(uniqid)

    # print(ptbk)

    # while ptbk is None or ptbk == '':
    #     ptbk = get_index_home(uniqid)
    all_data = data.get('all').get('data')
    result = decrypt(ptbk, all_data)
    result = result.split(',')

    # print('\n======= result ======\n')
    # print(result)
    return result

def demo():
    data = get_index_data(keyword='疫情拐点' ,start='2020-01-22' ,end='2020-05-31')
    for i in range(len(data)):
        if isinstance(data[i] , str) and len(data[i]) > 0:
            data[i] = int(data[i])
        else:
            data[i] = 0

    print(data)
    print(len(data))

    data = np.array(data)
    print(np.argmax(data))



if __name__ == '__main__':
    demo()

    data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5049, 6120, 5221, 6175, 4987, 6667, 5226,
            4062, 3714, 4242, 10526, 9309, 4793, 3461, 3916, 3940, 3951, 3957, 3679, 2779, 2747, 2160,
            1745, 1818, 1744, 1753, 1367, 1415, 1356, 1169, 1110, 1175, 1330, 1270, 1073, 795, 670, 510,
            533, 505, 533, 489, 451, 404, 351, 400, 405, 364, 366, 370, 391, 332, 305, 376, 387, 419, 431,
            344, 403, 367, 466, 450, 384, 370, 372, 309, 347, 434, 346, 369, 335, 333, 254, 270, 305, 326,
            320, 310, 282, 266, 271, 261, 300, 265, 240, 225, 226, 238, 225, 213, 237, 223, 239, 193, 203,
            214, 212, 216, 200, 177, 212, 211, 204, 203, 182, 184, 182, 160, 179, 170, 175, 171, 174, 173, 165, 169]