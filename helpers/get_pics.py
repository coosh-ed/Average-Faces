import urllib2
import sys
from bs4 import BeautifulSoup

from female_TDs import list_female_TDs

print list_female_TDs()

URL = "https://www.kildarestreet.com/tds/"

response = urllib2.urlopen(URL)
page_source = response.read()
soup = BeautifulSoup(page_source)

for img in soup.find_all('img'):
    print img['src']

