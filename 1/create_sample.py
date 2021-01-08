from random import randint

n_1,n_2=randint(1000,1200),randint(1000,1200)
d=["A","C","G","T"]
s_1,s_2="",""
for i in range(n_1):
    s_1=d[randint(0,3)]+s_1
for i in range(n_2):
    s_2=d[randint(0,3)]+s_2
f = open("data.txt", "w")
f.write("%s\n" %s_1)
f.write("%s\n" %s_2)
f.close()
