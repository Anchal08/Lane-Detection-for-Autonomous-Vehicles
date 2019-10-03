import json

# read file
file_path = "C:\\Users\\ganch\\Desktop\\Deep Learning\\Lane Detection\\bdd100k_labels_release\\bdd100k\\labels\\"
with open(file_path + str('bdd100k_labels_images_val.json'), 'r') as myfile:
    data=myfile.read()

# parse file
obj = json.loads(data)
print(len(obj))

obj1 = obj[1]
#print(str(obj1['labels'][4]['category']))

#print(len(obj1['labels']))
#print(obj1.keys())

info = []
d = {}

for k in range(len(obj)):
	vertex_list = []
	for label in obj[k]['labels']:
		if(label['category'] == 'lane'):
			for i in range(len(label['poly2d'])):
				for j in range(len(label['poly2d'][i]['vertices'])):
					tp = tuple(label['poly2d'][i]['vertices'][j])
					vertex_list.append(tp)

	d['name_'+str(k)] = obj[k]['name']
	d['vertex_'+str(k)] = vertex_list
	#info.append(d.copy())

print(d)

with open('data_val.json', 'w') as fp:
    json.dump(d, fp)
#import csv

# with open('Labels', 'wb') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerow(info)

#import numpy as np
#np.savetxt("Labels.csv", info, delimiter=",", fmt='%s')		

	
		

# show values
#print("usd: " + str(obj['usd']))
#print("eur: " + str(obj['eur']))
#print("gbp: " + str(obj['gbp']))