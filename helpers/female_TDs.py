# -*- coding: utf-8 -*-
import urllib2
from bs4 import BeautifulSoup

URL = "https://en.wikipedia.org/wiki/List_of_women_in_D%C3%A1il_%C3%89ireann"

def list_female_TDs():
    response = urllib2.urlopen(URL)
    page_source = response.read()
    soup = BeautifulSoup(page_source)

    tbl = soup.find('table')
    l = []
    for tr in tbl.find_all('tr', style="background-color: #cccccc"):
        TD = tr.td.a['title']
        if "(politician)" in TD:
            TD = " ".join(TD.split()[:-1])
        l.append(TD)

    return sorted(l)

if __name__=="__main__":
    print list_female_TDs()
