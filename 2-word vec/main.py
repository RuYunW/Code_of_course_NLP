import argparse
from utils import train
import time



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_path', help='txt file path', dest='fpath', default='./data/zh.txt', required=True)
    parser.add_argument('-model', help='Output model file', dest='save_path', default='./output/cbow_zh_vectors.txt', required=True)
    parser.add_argument('-cbow', help='1 for CBOW, 0 for skip-gram', dest='cbow', default=True, type=bool)
    parser.add_argument('-negative', help='Number of negative examples (>0) for negative sampling, 0 for hierarchical softmax', dest='neg', default=5, type=int)
    parser.add_argument('-dim', help='Dimensionality of word embeddings', dest='dim', default=100, type=int)
    parser.add_argument('-alpha', help='Starting alpha', dest='alpha', default=0.025, type=float)
    parser.add_argument('-window', help='Max window length', dest='win', default=5, type=int)
    parser.add_argument('-min-count', help='Min count for words used to learn <unk>', dest='min_count', default=5, type=int)
    args = parser.parse_args()

    start_time = time.clock()
    train(args.fpath, args.min_count, args.dim, args.neg, args.alpha, args.win, args.cbow, args.save_path)
    stop_time = time.clock()
    cost = stop_time - start_time
    print("totally cost %s second" % (cost))



    # train(args.fi, args.fo, bool(args.cbow), args.neg, args.dim, args.alpha, args.win,
    #       args.min_count, args.num_processes, bool(args.binary))
