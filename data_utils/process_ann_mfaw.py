import os

for filename in os.listdir('annotation'):
    if not filename.startswith('MAFW'):
        continue
    if filename.endswith('faces.txt'):
        continue
    with open('./annotation/'+filename, 'r') as f:
        a = f.readlines()
    corr = 0
    failed = 0
    for line in a:
        corr += 1
        path, amount, cla = line.split(' ')
        path = path.replace('datsets', 'datasets')
        path = path.replace('/data/EECS-IoannisLab/datasets/MAFW/data/faces', '/scratch/chumache/dfer_datasets/mfaw/clips_faces')
        amount2 = len(os.listdir('/scratch/chumache/dfer_datasets/mfaw/clips_faces/' + path.split('/')[-1]))
        if amount2 < 1:
            print(path, amount2)
            with open('dailed_mfaw.txt', 'a') as f:
               f.write(path + '\n')
        amount = amount2
        if not os.path.exists(path):
            continue
        with open('./annotation/'+filename.split('.')[0]+'_faces.txt', 'a') as f:
            f.write(path + ' ' + str(len(os.listdir(path))-1) + ' ' + cla)

