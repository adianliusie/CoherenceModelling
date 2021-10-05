import random

def create_corrupted_set(coherent, num_fake, schemes=[1], args=[1]):
    corrupted_set = []
    fail = 0
    while len(corrupted_set) < num_fake and fail<50:
        _, incoherent = create_incoherent(coherent, schemes, args)
        if (incoherent not in corrupted_set) and (incoherent != coherent): corrupted_set.append(incoherent)
        else: fail += 1
    return corrupted_set

def create_incoherent(coherent, schemes, args):
    r = random.choice(schemes)
    if   r == 1: incoherent = random_shuffle(coherent)
    elif r == 2: incoherent = random_swaps(coherent, args[0])
    elif r == 3: incoherent = random_neighbour_swaps(coherent, args[0])
    elif r == 4: incoherent = random_deletion(coherent, args[0])
    elif r == 5: incoherent = local_word_swaps(coherent, *args)
    else: raise Exception
    return coherent, incoherent

def random_shuffle(conversation):
    incoherent = conversation.copy()
    random.shuffle(incoherent)
    return incoherent

def random_swaps(conversation, num_swaps=1):
    incoherent = conversation.copy()
    indices = random.sample(range(0, len(incoherent)), min(2*num_swaps, 2*int(len(incoherent)/2)))

    for i in range(0, len(indices), 2):
        ind_1, ind_2 = indices[i], indices[i+1]
        incoherent[ind_1], incoherent[ind_2] = incoherent[ind_2], incoherent[ind_1]
    return incoherent

def random_neighbour_swaps(conversation, num_swaps=1):
    incoherent = conversation.copy()
    indices = random.sample(range(1, len(incoherent)), num_swaps)
    for i in indices:
        incoherent[i], incoherent[i-1] = incoherent[i-1], incoherent[i]
    return incoherent

def random_deletion(conversation, num_delete=1):
    incoherent = conversation.copy()
    _ = [conversation.pop(-1) for i in range(num_delete)]
    indices = random.sample(range(1, len(incoherent)-1), num_delete)
    indices.sort(reverse=True)
    for i in indices:
        incoherent.pop(i)
    return incoherent

def local_word_swaps(conversation, num_sents=1, num_word_swaps=1):
    incoherent = conversation.copy()
    indices = random.sample(range(0, len(incoherent)), num_sents)
    for i in indices:
        words = incoherent[i].split()
        positions = random.sample(range(0, len(words)), min(2*num_word_swaps, 2*(len(words)//2)))
        for j in range(0, len(positions), 2):
            ind_1, ind_2 = positions[j], positions[j+1]
            words[ind_1], words[ind_2] = words[ind_2], words[ind_1]
        sentence = ' '.join(words)
        incoherent[i] = sentence
    return incoherent

