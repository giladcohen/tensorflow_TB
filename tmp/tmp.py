SET_SIZE = 30
C_SIZE = 3

p = []
available_samples = range(SET_SIZE)

for i in range(10):
    p_tmp = random.sample(available_samples, C_SIZE)
    p += p_tmp
    p = sorted(p)
    available_samples = [i for j, i in enumerate(available_samples) if i not in p]
    print ('p_tmp =')
    print (p_tmp)
    print ('available_samples = ')
    print (available_samples)
    print ('p = ')
    print (p)