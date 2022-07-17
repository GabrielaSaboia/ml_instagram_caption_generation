import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from ml_instagram_caption_generation.backend.caption_model import predict_caption


def bleu_score(img_text_model, fnm_test, di_test, dt_test, tokenizer,
               index_word):
    nkeep = 5
    pred_good, pred_bad, bleus = [], [], []
    count = 0
    for jpgfnm, image_feature, tokenized_text in zip(fnm_test, di_test,
                                                     dt_test):
        count += 1
        if count % 200 == 0:
            print("  {:4.2f}% is done..".format(
                100 * count / float(len(fnm_test))))

        caption_true = [index_word[i] for i in tokenized_text]
        caption_true = caption_true[1:-1]
        # captions
        caption = predict_caption(img_text_model,
                                  image_feature.reshape(1, len(image_feature)),
                                  tokenizer,
                                  index_word)
        caption = caption.split()
        caption = caption[1:-1]

        bleu = sentence_bleu([caption_true], caption)
        bleus.append(bleu)
        if bleu > 0.7 and len(pred_good) < nkeep:
            pred_good.append((bleu, jpgfnm, caption_true, caption))
        elif bleu < 0.3 and len(pred_bad) < nkeep:
            pred_bad.append((bleu, jpgfnm, caption_true, caption))

    print('Mean BLEU {:4.3f}'.format(np.mean(bleus)))
