from glob import glob
from os.path import join

train_scenes = [x.split('/')[-1][:-15] for x in glob('train/*.pth')]

with open('list/train_2D.txt','w') as f:
	for s in train_scenes:
		for im in glob(join('2D',s,'color','*.jpg')):
			f.write(im+' '+im.replace('color','label').replace('.jpg','.png')+'\n')

	
val_scenes = [x.split('/')[-1][:-15] for x in glob('val/*.pth')]

with open('list/val_2D.txt','w') as f:
	for s in val_scenes:
		for im in glob(join('2D',s,'color','*.jpg')):
			f.write(im+' '+im.replace('color','label').replace('.jpg','.png')+'\n')
