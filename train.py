from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    #model.restore_session(config.dir_model)
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)

import os

if __name__ == "__main__":

    if os.name != 'nt':  # not windows
        print('current working dir [{0}]'.format(os.getcwd()))
        w_d = os.path.dirname(os.path.abspath(__file__))
        print('change wording dir to [{0}]'.format(w_d))
        os.chdir(w_d)

    main()
