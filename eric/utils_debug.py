def segment(sent, step=3, wind=[100, 70, 40]):
    """

    :param corp: corpus from which the segment are extracted
    :param step:
    :param wind: the lenghts of the differents segments extracted at a same time
    :return: a stack of embeded segments and a reference file
    """
    segs = []
    seg_ref = {}
    cpt = 0
    utt_lengths = []
    i_start = 0
    w1 = []
    w07 = []
    w04 = []
    n_search = sent.shape[0]
    while i_start <= n_search - 1 or i_start == 0:
        for ind in wind:
            if ind == 100:
                w1.append(cpt)
            elif ind == 70:
                w07.append(cpt)
            elif ind == 40:
                w04.append(cpt)
            sref = {}
            sref["wind"] = ind
            sref["time"] = i_start/100
            seg_ref[cpt] = sref
            seg=sent[i_start:i_start+ind]
            utt_lengths.append(seg.shape[0])
            i_start += step
            cpt +=1
            segs.append(seg)
    return segs, utt_lengths, w1, w04, w07