def mapValue(x):
    if x=="A":
        return 0
    elif x=="B":
        return 1
    elif x=="BC":
        return 2
    elif x=="C":
        return 3
    elif x=="CD":
        return 4
    elif x=="D":
        return 5
    elif x=="DE":
        return 6
    elif x=="E":
        return 7
    elif x=="EF":
        return 8
    else:
        return 9

l= ['A','BC','DE','EF','B','A','C']
for i in range(len(l)):
    l[i]=mapValue(l[i])
    i+=1

    