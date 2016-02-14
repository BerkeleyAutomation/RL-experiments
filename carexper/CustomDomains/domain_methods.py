def allMarkovEncoding(ps):
    return [0]

def stateVisitEncoding(ps, waypoints):
    result = []
    for w in waypoints:
        k = -1
        pl = [tuple(p) for p in ps]
        try:
            k = pl.index(w)
        except ValueError:
            pass
        result.append(k)

    result_hash = []

    if len(waypoints) == 1 and result[0] == -1:
        return [0]
    elif len(waypoints) == 1 and result[0] != -1:
        return [1]
    
    if result[0] != -1:
        result_hash.append(1)
    else:
        result_hash.append(0)

    for i in range(1,len(waypoints)):
        if result[i] != -1 and result[i] > result[i-1]:
            result_hash.append(1)
        else:
            result_hash.append(0)
    #print len(waypoints), result_hash
    return result_hash

"""
Popular reward functions that you could use
"""
def allMarkovReward(ps,ga, sr, gr):
    r = sr
    last_state = ps[len(ps)-1][0]
    if (last_state[0],last_state[1]) in ga:
        r = gr
    return r

implicitReward = None