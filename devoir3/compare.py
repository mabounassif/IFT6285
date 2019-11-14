dicts = {}
f = open("compare.output", "w")

for i in range(3):
    dicts[i] = {}
    fp = open('class_%s_freq_table'%i, 'r')
    for l in fp:
        split = l.split()
        if len(split) != 2:
            continue 

        count, word = split
        dicts[i][word] = int(count)

for i in range(3):
    for word in dicts[i]:
        other_count = sum([dicts[j][word] if word in dicts[j] else 0 for j in set(range(3)) - { i }])
        perc = (dicts[i][word] - other_count) / dicts[i][word]

        if perc > 0.8 and dicts[i][word] >= 100:
            f.write('%s %s %s %s %s\n'%(perc, i, dicts[i][word], other_count, word))

f.close()
                
