from nltk.translate.bleu_score import corpus_bleu

f_truth = open("truth.txt", "r", encoding="utf-8")
f_pred = open("pred.txt", "r", encoding="utf-8")


reference = []
candidate = []
for truth, pred in zip(f_truth, f_pred):
	truth = truth.strip().split(" ")[1: -1]
	pred = list(pred.strip())
	reference.append([truth])
	candidate.append(pred)

print('Individual 1-gram: %f' % corpus_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Individual 2-gram: %f' % corpus_bleu(reference, candidate, weights=(0, 1, 0, 0)))
print('Individual 3-gram: %f' % corpus_bleu(reference, candidate, weights=(0, 0, 1, 0)))
print('Individual 4-gram: %f' % corpus_bleu(reference, candidate, weights=(0, 0, 0, 1)))
