file_path = './code/maincode.py'
save_path = './code/maincode.tex'

with open(file_path, 'r') as f:
    lines = f.readlines()

with open(save_path, 'w') as ff:
    ff.write('\\begin{python}\n')
    for line in lines:
        line.replace('\t', '    ')
        ff.write(line)
    ff.write('\\end{python}\n')

