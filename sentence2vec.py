# coding: utf-8
from gensim.models import KeyedVectors
import numpy as np
import MeCab
from chainer import functions as F
import re
import neologdn
from gensim import corpora
from sklearn.decomposition import PCA
import sys

corpusdir = ""
file_w2v_hottolink = corpusdir + ""

mt = MeCab.Tagger('')
model_hottolink = KeyedVectors.load_word2vec_format(file_w2v_hottolink, binary=False)


def wakati_mecab(text):
    result = []
    for morph in mt.parse(neologdn.normalize(text)).rstrip().split('\n')[:-1]:
        surface = re.split(r',|\t', morph)[0]
        result.append(surface)
    return result


class SWEM():
    def __call__(self, text, mode):
        vecs = self.w2v(text)
        if mode == "aver":
            return vecs.mean(axis=0)
        elif mode == "max":
            return vecs.max(axis=0)
        elif mode == "concat":
            return np.hstack([vecs.mean(axis=0), vecs.max(axis=0)])
        elif mode == "hier_2":
            return self.hier(vecs, 2)

    def w2v(self, text):
        sep_text = wakati_mecab(text)
        v = []
        print(sep_text)
        for w in sep_text:
            try:
                v.append(model_hottolink[w])
            except KeyError:
                v.append(np.zeros(200))
        if not v:
            v.append(np.zeros(200))
        return np.array(v)

    def hier(self, vecs, window):
        h, w = vecs.shape
        if h < window:
            return vecs.max(axis=0)
        v = F.average_pooling_nd(vecs.reshape(1, w, h), ksize=window).data
        return v.max(axis=2)[0]


def get_swem_vector(text, texts, sentence_embeding_method="aver"):
    swem = SWEM()
    swem_hier_2_vecs = np.array(swem(text, "hier_2"))
    return swem_hier_2_vecs


class SIF():
    def sif_vector(self, texts, sentence_embeding_method="aver"):
        docs = []
        for text in texts:
            docs.append(wakati_mecab(text))
        dictionary = corpora.Dictionary(docs)
        w2pw = {}  # 単語の出現確率
        for k in dictionary.token2id.keys():
            w2pw[k] = dictionary.cfs[dictionary.token2id[k]] / dictionary.num_pos
        a = 10 ** -3  # sif のパラメータ定数
        sentence_vectors = []
        if sentence_embeding_method == "aver":
            sentence_vectors = self.mean_vector(docs, texts, w2pw, a)
        elif sentence_embeding_method == "hier":
            sentence_vectors = self.hier_vector(docs, texts, w2pw, a, 2)
        if sentence_vectors == []:
            print("choice embeding method [mean, hier]")
            sys.exit()
        pca = PCA(n_components=1)
        pca.fit_transform(sentence_vectors)
        c0 = pca.components_[0]
        for i in range(len(sentence_vectors)):
            sentence_vectors[i] -= (c0 * (c0.T @ sentence_vectors[i])).T
        dictionary.save_as_text('deerwester_dict.txt')
        return sentence_vectors

    def mean_vector(self, docs, texts, w2pw, a):
        sentence_vectors = []
        for doc in docs:
            sum_vec = np.zeros(200)
            for w in doc:
                try:
                    sum_vec += (a / (a + w2pw[w]) * model_hottolink[w])
                except KeyError:
                    pass
            if len(doc) >= 1:
                sentence_vectors.append(sum_vec / len(doc))
            else:
                sentence_vectors.append(np.zeros(200))
        return np.array(sentence_vectors)

    def hier_vector(self, docs, texts, w2pw, a, window):
        sentence_vectors = []
        for doc in docs:
            vecs = []
            for w in doc:
                try:
                    vecs.append(a / (a + w2pw[w]) * model_hottolink[w])
                except KeyError:
                    vecs.append(np.zeros(200))
            if vecs == []:
                vecs.append(np.zeros(200))
            vecs = np.array(vecs)
            h, w = vecs.shape
            if h < window:
                sentence_vector = vecs.max(axis=0)
            else:
                v = F.average_pooling_nd(vecs.reshape(1, w, h), ksize=window).data
                sentence_vector = v.max(axis=2)[0]
            sentence_vectors.append(sentence_vector)
        return np.array(sentence_vectors)


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def run():
    sif = SIF()
    texts = [
        "北海道行きたい",
        "北海道に帰ろう"
    ]
    vec = sif.sif_vector(texts, "hier")
    print(cos_sim(vec[0], vec[0]))


if __name__ == "__main__":
    run()
