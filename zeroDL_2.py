
def AND(x1,x2):
    w1,w2,d = 0.5,0.5,0.7
    E = -d + w1*x1 + w2*x2
    if E > 0:
        E1 = 1
    elif E < 0:
        E1 = 0
    print("AND:",E1)

AND(1,1)
AND(1,0)
AND(0,1)
AND(0,0)

def OR(x1,x2):
    w1,w2,d = 0.5,0.5,0.5
    E = -d + w1*x1 + w2*x2
    if E >= 0:
        E1 = 1
    elif E < 0:
        E1 = 0
    print("OR:",E1)

OR(1,1)
OR(1,0)
OR(0,1)
OR(0,0)

def NAND(x1,x2):
    w1,w2,d = -0.5,-0.5,0.7
    E = d + w1*x1 + w2*x2
    if E > 0:
        E1 = 1
    elif E < 0:
        E1 = 0
    print("AND:",E1)

NAND(1,1)
NAND(1,0)
NAND(0,1)
NAND(0,0)
