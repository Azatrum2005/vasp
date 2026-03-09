import numpy as np

# Read your POSCAR
with open('VASPfiles/Bi2Se3_3QL.vasp', 'r') as f:
    lines = f.readlines()

# # Change third lattice vector
# lines[4] = '        0.0000000000         0.0000000000        20.0000000000\n'

z=0
# for i in range(8, 8+8) and [8,10,13,15]: 
#     print(i)
#     coords = lines[i].split()
#     z += float(coords[2])
# a=z/4
# print(a)
# shift = 7.5-a

for i in range(8, 8+15): 
    coords = lines[i].split()
    z = 10 + float(coords[2])
    # if i in [12,17,20]:
    #     z = 30.4111518860 + float(coords[2]) #- 6.625160672 #shift
    # else:
    #     z = float(coords[2])
    lines[i] = f'    {coords[0]}    {coords[1]}    {z}\n'
print(lines)

with open('Bi2Se3_3QL.vasp', 'w') as f:
    f.writelines(lines)