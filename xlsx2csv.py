import os
import xlrd
import csv

def csv_from_excel(of, nf):

    print (of, nf)

    wb = xlrd.open_workbook(of)
    sh = wb.sheet_by_name('Sheet1')
    your_csv_file = open(nf, 'wb')
    wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

    for rownum in xrange(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    your_csv_file.close()

dirname = './Leakages'
new_ext = ".csv"
fileList = []
for fn in os.listdir(dirname):
    fn1 = os.path.join(dirname, fn)
    if os.path.isfile(fn1):
        fileList.append(fn1)
print "\n".join(fileList)
     
output = []
for fn in fileList:
    dn, fn1 = os.path.split(fn)
    nf = os.path.join(dn, os.path.splitext(fn1)[0]+new_ext)
    output.append("Old file name: %s\nNew file name: %s\n" %
                (fn, nf))
    if len(fn) < 4 or fn[-4:] != "xlsx":
        continue
    csv_from_excel(fn, nf)
 # os.sa
# print "\n".join(output)