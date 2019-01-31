import numpy as np
from keras.preprocessing.sequence import pad_sequences

if __name__ == '__main__':
    # print(np.random.choice(list('0123456789')))
    # print(np.random.randint(1, 4))
    x = [[i+1] for i in range(4) if i == 1]
    print(pad_sequences([[1, 9, 3], [1, 2, 3, 4, 5]], maxlen=4))
    # l = np.zeros((3, 3, 3), dtype=np.bool)
    # l1 = l[np.array([0])]
    # l1[0][0][1] = 1
    # l1[0][1][0] = 1
    # l1[0][2][0] = 1
    # # print(l1[0])
    # l1 = l1[0]
    # x = l1.argmax(axis=-1)
    # nid, line = "2 Mary got the milk there.".split(" ", 1)
    # print(nid)
    # print(line)