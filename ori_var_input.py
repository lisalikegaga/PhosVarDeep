import csv
import time
import os

#get variant sequence


filename = './testdata.csv'
#var = []
fw1 = open('./testdata_ori.csv', 'w')
fw2 = open('./testdata_var.csv', 'w')

with open(filename, 'r') as rf:
    reader = csv.DictReader(rf)
    for row in reader:
       l =  list(row['mutation'])
       ori = l[0]
       var = l[-1]
       num1 = "".join(l[1:-1])
       num = int(num1)
       ori_seq = list(row['ori_seq'])
       pos = int(row['pos'])
       if len(ori_seq) < pos or len(ori_seq) < num:
           var_seq = '0'
       else:
         if ori_seq[num-1] == ori:
            ori_seq[num-1] = var
            var_seq = "".join(ori_seq)
         else:
           var_seq = '0'

       line = row['label'] + ',' + row['protein'] + ',' + row['pos'] + ',' + row['ori_seq'] + '\n'
       fw1.write(line)
       line = row['label'] + ',' + row['protein'] + ',' + row['pos'] + ',' + str(var_seq) + '\n'
       fw2.write(line)

fw1.close()
fw2.close()
rf.close()




