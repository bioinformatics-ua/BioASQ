# This is a file to test some python code, does not belong with the project
from utils import LimitedDict

a = LimitedDict(10)


for i in range(20):
    a[i] = chr(i+67)

print(a[10])
print(a.frequency_list)
print(len(a), a.current_elments)
