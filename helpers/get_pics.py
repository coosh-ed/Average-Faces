import urllib, urllib2, json
import sys, os
from bs4 import BeautifulSoup

from female_TDs import list_female_TDs


def get_image_url(name):
    header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
    name = name.encode('utf-8').strip()
    keyword = name + " TD"
    query = "+".join(keyword.split())
    url_googleimages = "https://www.google.ie/search?q="+query+\
"&source=lnms&tbm=isch&sa=X&ved=0ahUKEwii1tyIgNrWAhWKJVAKHRHIDmMQ_AUICigB&biw=1280&bih=624" 
    urlopen = urllib2.urlopen(urllib2.Request(url_googleimages,headers=header))
    soup = BeautifulSoup(urlopen,'html.parser')
    src = json.loads(soup.find("div",{"class":"rg_meta"}).text)['ou']
    return src


names = list_female_TDs()

#create folders to save pictures
if not os.path.exists('maleTDs'):
    os.makedirs('maleTDs')
if not os.path.exists('femaleTDs'):
    os.makedirs('femaleTDs')


URL = "https://www.kildarestreet.com/tds/"

response = urllib2.urlopen(URL)
page_source = response.read()
soup = BeautifulSoup(page_source)

tbl = soup.tbody

for tr in soup.find_all('tr'):
    name = tr.find('a').text
    x = "female" if name in names else "male" 
    
    img_url = get_image_url(name)
    urllib.urlretrieve(img_url, x+"TDs/"+os.path.basename(img_url))
    

