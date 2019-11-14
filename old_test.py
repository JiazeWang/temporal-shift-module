import numpy as np
def class_precision(scores, labels):
	sortidx = np.argsort(-scores)
	tp = (labels[sortidx]==1).astype(int)
	fp = (labels[sortidx]!=1).astype(int)
	npos = labels.sum()
	fp=np.cumsum(fp)
	tp=np.cumsum(tp)
	rec=tp/npos
	prec=tp/(fp+tp)

	ap = 0
	tmp = (labels[sortidx]==1).astype(int)
	for i in range(len(scores)):
		if tmp[i]==1:
			ap=ap+prec[i];
	ap=ap/npos
	return ap

def compute_map(labels, test_scores):
    nclasses = labels.shape[1]
    if nclasses != 21:
        print ('class num wrong! ')
        sys.exit()
    ap_all = np.zeros(labels.shape[1])
    for i in range(nclasses):
        ap_all[i] = class_precision(test_scores[:, i], labels[:, i])
    mAP = np.mean(ap_all)
    #print(ap_all)
    #print(mAP)
    wAP = np.sum(ap_all*np.sum(labels,0))/np.sum(labels);
    return mAP, wAP
predict = np.load("val.npy")
label = np.load("movie_val.npy")
m1,w1 = compute_map(label, predict) 
