# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import json
import urllib
import requests
from urllib.parse import urlparse
import httplib2 as http #External library


if __name__=="__main__":
    #Authentication parameters
    headers = { 'AccountKey' : 'NbWuSByQSWq20xkWUveHag==',
               'accept' : 'application/json'} #this is by default
    
    #API parameters
    uri = 'http://datamall2.mytransport.sg/' #Resource URL
    URLs = []
    for months in range(5, 8):
        path = 'ltaodataservice/PV/Train?Date=20200' + str(months)
        #Build query string & specify type of API call
        target = urlparse(uri + path)
        
        print (target.geturl())
        method = 'GET'
        body = ''
        
        #Get handle to http
        h = http.Http()
        #Obtain results
        response, content = h.request(
            target.geturl(),
            method,
            body,
            headers)
        #Parse JSON to print
        jsonObj = json.loads(content)
        print(json.dumps(jsonObj, sort_keys=True, indent=4))
        URLs += jsonObj['value']
    #Save result to file
    with open("Train.json","w") as outfile:
        #Saving jsonObj["d"]
        json.dump(URLs, outfile, sort_keys=True, indent=4,
                  ensure_ascii=False)
        
        
        
        
        
        
        