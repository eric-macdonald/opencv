#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python
import os
import sys
import fileinput

fileToSearch = 'temp'
textToSearch = '0515'
tempFile = open( fileToSearch, 'r+' )
index = 1000
for line in fileinput.input( fileToSearch ):
        if textToSearch in line :
	   line = line.replace(textToSearch, str(index))
	   line = line.strip('\n')
           print line
	   index = index + 1
tempFile.close()
