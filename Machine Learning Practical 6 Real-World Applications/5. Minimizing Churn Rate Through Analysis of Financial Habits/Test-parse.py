import xml.etree.ElementTree
from xml.dom.minidom import parseString

data = ''
with open('AndroidManifest.xml','r') as f:
    data = f.read()
dom = parseString(data)
nodes = dom.getElementsByTagName('uses-permission')
# Iterate over all the uses-permission nodes
for node in nodes:
    #print(node.toxml())
    print(node.getAttribute('android:name'))


#e = xml.etree.ElementTree.parse('AndroidManifest.xml').getroot()


