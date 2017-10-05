import urllib2

response = urllib2.urlopen("")
page_source = response.read()
