def ReadData(str):
    f = open(str,'r')
    data = []
    for line in f:
        current = line.split(',')
        data.append(current)
    f.close()

    return data

def write_data(data):
    f1 = open('dataset/wdbc_data.txt', 'w')
    f2 = open('dataset/wdbc_label.txt','w')
    for i in range(len(data)):
        item = [data[i][j] for j in range(2,len(data[i]))]
        label = data[i][1]
        dummy = ','.join(item)
        print(dummy,file=f1,end = '')
        print(label,end='\n',file=f2)
    f1.close()
    f2.close()

data = ReadData('dataset/wdbc.data')
write_data(data)