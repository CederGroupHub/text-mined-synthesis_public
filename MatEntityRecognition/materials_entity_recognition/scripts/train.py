# %%
# Some preparation: load libraries, define functions
# %
# %%
import argparse
import os
import sys
parent_folder = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..'
    )
)
print('parent_folder', parent_folder)
if parent_folder not in sys.path:
    sys.path.append(parent_folder)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in {'yes', 'true', 't', 'y', '1'}:
        return True
    elif v.lower() in {'no', 'false', 'f', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2None(v):
    if v is None:
        return v
    if v.lower() in {'none', }:
        return None
    return v

def get_argument_parser():
    # Read parameters from command line
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--tag_scheme",
        help="Tagging scheme (iob or iobes)",
        default='iobes',
    )
    argparser.add_argument(
        "--clean_tag", type=str2bool,
        help="If True, neglect 'I-*' and 'E-*' tags when 'B-*' tag is not found",
        default=True
    )
    argparser.add_argument(
        "--lower", type=str2bool,
        help="Lowercase words",
        default=False
    )
    argparser.add_argument(
        "--zeros", type=str2bool,
        help="Replace digits with 0",
        default=False
    )
    argparser.add_argument(
        "--classifier_type",
        help="Classifier applied to final inputs after embedding layers",
        default='lstm',
    )
    argparser.add_argument(
        "--bert_path", type=str2None,
        help="Path to pretained bert model",
        default=None,
    )
    argparser.add_argument(
        "--bert_first_trainable_layer", type=int,
        help="First bert encoder layer which is trainable, range [-12, 12]."
             "-12 or 0 means all encoder layers are trainable."
             "12 means no layer is trainable."
             "Trainable layers are [bert_first_trainable_layer:12]",
        default=6
    )
    argparser.add_argument(
        "--word_dim", type=int,
        help="Token embedding dimension",
        default=100
    )
    argparser.add_argument(
        "--word_lstm_dim", type=int,
        help="Token LSTM hidden layer dimension",
        default=100
    )
    argparser.add_argument(
        "--word_bidirect", type=str2bool,
        help="Use a bidirectional LSTM for words",
        default=True
    )
    argparser.add_argument(
        "--word_unroll", type=str2bool,
        help="Use unroll or not in LSTM for words",
        default=False
    )
    argparser.add_argument(
        "--word_rnn_wrapper", type=str2bool,
        help="Use rnn_wrapper or not in LSTM for words",
        default=False
    )
    argparser.add_argument(
        "--char_dim", type=int,
        help="Character embedding dimension",
        default=25
    )
    argparser.add_argument(
        "--char_lstm_dim", type=int,
        help="Character LSTM hidden layer dimension",
        default=25
    )
    argparser.add_argument(
        "--char_bidirect", type=str2bool,
        help="Use a bidirectional LSTM for characters",
        default=True
    )
    argparser.add_argument(
        "--char_combine_method",
        help="Method (concat/sum) used to combine forward and backward "
             "char embeddings. Only used when char_bidirect is True ",
        default='concat'
    )
    argparser.add_argument(
        "--char_unroll", type=str2bool,
        help="Use unroll or not in LSTM for chars",
        default=False
    )
    argparser.add_argument(
        "--char_rnn_wrapper", type=str2bool,
        help="Use rnn_wrapper or not in LSTM for chars",
        default=False
    )
    argparser.add_argument(
        "--use_ori_text_char", type=str2bool,
        help="Use original characters rather than simple language in character layer",
        default=False
    )
    argparser.add_argument(
        "--use_ele_num", type=str2bool,
        help="Use element number as a feature or not",
        default=False
    )
    argparser.add_argument(
        "--use_only_CHO", type=str2bool,
        help="Use only_CHO as a feature or not",
        default=False
    )
    argparser.add_argument(
        "--rnn_type",
        help="Type of RNN (lstm/gru)",
        default='gru'
    )
    argparser.add_argument(
        "--crf", type=str2bool,
        help="Use CRF or not",
        default=True
    )
    argparser.add_argument(
        "--crf_begin_end", type=str2bool,
        help="Use begin/end tag in CRF layer or not",
        default=True
    )
    argparser.add_argument(
        "--dropout", type=float,
        help="Droupout on the input (0 = no dropout)",
        default=0.5
    )
    argparser.add_argument(
        "--lr_method",
        help="Learning method (SGD, Adadelta, Adam..) and "
             "learning rate (0.005 as default)"
             "E.g: sgd@lr=0.005, adam@lr=0.00005, adamdecay@lr=5e-05@epsilon=1e-08@warmup=0.1",
        default='sgd@lr=0.005'
    )
    argparser.add_argument(
        "--loss_per_token", type=str2bool,
        help="Average loss by number of tokens in each sentence or not",
        default=False
    )
    argparser.add_argument(
        "--batch_size", type=int,
        help="batch_size",
        default=1
    )
    argparser.add_argument(
        "--num_epochs", type=int,
        help="number of epochs",
        default=100
    )
    argparser.add_argument(
        "--steps_per_epoch", type=int,
        help="steps_per_epoch",
        default=3500
    )
    argparser.add_argument(
        "--training_gen_mode",
        help="mode (static/dynamic) used to generate training set "
             "when dynamic mode is used, an explicit integer should "
             "be assigned to steps_per_epoch ",
        default='dynamic'
    )
    argparser.add_argument(
        "--training_ratio", type=float,
        help="fraction of sentences to use in training set",
        default=1.0
    )
    argparser.add_argument(
        "--path_train",
        help="path_train",
        default='dataset/TP_750_1/TP_750_1_00/train.json'
    )
    argparser.add_argument(
        "--path_dev",
        help="path_dev",
        default='dataset/TP_750_1/TP_750_1_00/dev.json'
    )
    argparser.add_argument(
        "--path_test",
        help="path_test",
        default='dataset/TP_750_1/TP_750_1_00/test.json'
    )
    argparser.add_argument(
        "--emb_path", type=str2None,
        help="path to embedding file",
        default='dataset/embedding/embedding_sg_win5_size100_iter50_noLemma.text'
    )
    argparser.add_argument(
        "--word_dico_source", type=str2None,
        help="source of word dico (train/train_dev_test) ",
        default='train_dev'
    )
    argparser.add_argument(
        "--singleton_unk_probability", type=float,
        help="randomly convert singleton to <UNK> in training by p",
        default=0.5
    )
    argparser.add_argument(
        "--test_fake_tar_tag", type=str2bool,
        help="test 0-1 feature with fake target tag",
        default=False
    )
    argparser.add_argument(
        "--test_fake_pre_tag", type=str2bool,
        help="test 0-1 feature with fake precurosr tag",
        default=False
    )
    argparser.add_argument(
        "--to_reload_model", type=str2bool,
        help="set true if reloading model for prediction "
             "to load embeddings is used. deprecated",
        default=False
    )
    argparser.add_argument(
        "--model_dir", type=str2None,
        help="path to save checkpoint files and model files. "
             "should be specified if reloading pre-trained model",
        default='generated/model_1'
    )
    argparser.add_argument(
        "--std_out",
        help="use screen or file as stdout ",
        default="screen"
    )
    argparser.add_argument(
        "--device",
        help="use gpu or cpu ",
        default="gpu"
    )
    return argparser

def parse_argument():
    # Read parameters from command line
    argparser = get_argument_parser()
    args = argparser.parse_args()
    return args

#########################################
# use gpu or cpu
#########################################
args_0 = parse_argument()
if args_0.device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import datetime
import shutil

from scripts import loader
from scripts import utils
from scripts import callbacks
from scripts import model_framework

import tensorflow as tf
from tensorflow import keras
if utils.found_package('transformers'):
    import transformers

def validate_parameters(parameters):
    """
    Make sure the parameters are valid
    """
    assert parameters.word_dim >= 0
    assert 0. <= parameters.dropout < 1.0
    assert parameters.tag_scheme in ['iob', 'iobes']
    if parameters.bert_path:
        assert parameters.word_dim == 0
        assert parameters.char_dim == 0
        assert parameters.emb_path == None
        assert parameters.singleton_unk_probability == 0.0


if __name__ == '__main__':
    #########################################
    # paramter specification
    #########################################
    args = parse_argument()
    if args.std_out != 'screen':
        utils.use_file_as_stdout(args.std_out)
    utils.print_gpu_info()

    print('start time', datetime.datetime.now())
    print('parameters: ', args.__dict__)
    # derived
    # checkpoint path
    args.model_dir = os.path.abspath(args.model_dir)
    if not args.to_reload_model:
        if os.path.exists(args.model_dir):
            shutil.rmtree(args.model_dir)
        os.makedirs(args.model_dir)

    # data loading
    #########################################
    # load data
    #########################################
    # basically, you can use any format you like to load data
    # here is an example for CoNLL format
    # The training/validation/test data file consists of many lines,
    # each of which is a token and its attributes separated by blanks,
    # e.g.: word O Y tag
    # the first entry is the token text
    # the last entry is the tag, such as B-Mat (beginning of material),
    # I-Tar (intermediate parte of target), O (outside)
    # other entries between the first and the last can be used to feed pre-engineered features,
    # which are not used in this example
    # There is also empty lines, which are used to separate tokens in different sentences
    # tokens between two empty lines form a sentence

    # embedding loading
    # There are three ways to load embeddings
    # 1. use a file, in which each line is a token and its embeddings,
    #  e.g.: word 0.0 0.0 0.0 ... 0.0
    # specify emb_path in this manner
    # 2. use a dict (like word2vec in gensim)
    # specify emb_dict in this manner
    # 3. use a matrix (numpy 2d array), the index of matrix should be consist with the vocab provided
    # specify params['pre_emb'] in this manner

    # provide the vocab if you want to use all words in embedding source
    # in this example, vocab is [] and later the words in the whole dataset
    # (including training/validation/test sets) are assigned to vocab,
    # because embedding file is provided
    # <UNK> is automatically inserted as the first token in vocab

    # ---------------not need to be edited if not changing the format of data files-----------------------------
    # Data parameters

    # Load sentences
    train_sentences = loader.load_sentences(
        args.path_train, args.lower, args.zeros
    )
    val_sentences = loader.load_sentences(
        args.path_dev, args.lower, args.zeros
    )
    test_sentences = loader.load_sentences(
        args.path_test, args.lower, args.zeros
    )

    # Use selected tagging scheme (IOB / IOBES)
    loader.update_tag_scheme(train_sentences, args.tag_scheme)
    loader.update_tag_scheme(val_sentences, args.tag_scheme)
    loader.update_tag_scheme(test_sentences, args.tag_scheme)

    # Create a dictionary and a mapping for words / POS tags / tags
    dico_tags, tag_to_id, id_to_tag = loader.tag_mapping(
        train_sentences
    )

    if args.bert_path:
        config_template = 'bert-base-cased'
        bert_tokenizer = transformers.BertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=args.bert_path,
            model_max_length=transformers.BertTokenizerFast.max_model_input_sizes[config_template],
            **transformers.BertTokenizerFast.pretrained_init_configuration[config_template]
        )
        vocab = None
        word_to_id = None
        id_to_word = None
        singletons = None
        dico_chars = None
        char_to_id = None
        id_to_char = None
        pre_emb = None
    else:
        # Create a dictionary / mapping of words
        # If we use pretrained embeddings, we add them to the dictionary.
        if (args.word_dico_source == 'train_dev_test'):
            vocab, word_to_id, id_to_word = loader.word_mapping(
                train_sentences + val_sentences + test_sentences, args.lower
            )
        elif (args.word_dico_source == 'train_dev'):
            vocab, word_to_id, id_to_word = loader.word_mapping(
                train_sentences + val_sentences, args.lower
            )
        elif (args.word_dico_source == 'train'):
            vocab, word_to_id, id_to_word = loader.word_mapping(
                train_sentences, args.lower
            )

        # singletons are words only appear one time in training set
        dico_words_train, _, _ = loader.word_mapping(
                train_sentences, args.lower
            )
        singletons = set([word_to_id[k] for k, v
                          in list(dico_words_train.items()) if v == 1])

        # Create a dictionary and a mapping for chars
        dico_chars, char_to_id, id_to_char = loader.char_mapping(
            train_sentences,
            use_ori_text_char=args.use_ori_text_char
        )

        # get embedding matrix
        if args.emb_path and args.word_dim:
            pre_emb = loader.prepare_embedding_matrix(
                id_to_word,
                args.word_dim,
                emb_path=args.emb_path,
            )
        else:
            pre_emb = None
        bert_tokenizer = None

    # ensure thee parameters are valid
    validate_parameters(args)

    #########################################
    # prepare NN input
    #########################################
    # "Token embedding dimension"
    # the words in sentences are automatically converted to
    # a list such as [ {'words': [w0, w1, w2]} , ... ],
    # where each element is the id of word (index in one-hot vector)
    # then the embedding is automatically loaded

    train_X, train_Y, train_data, train_sentences = loader.prepare_dataset(
        sentences=train_sentences,
        word_to_id=word_to_id,
        char_to_id=char_to_id,
        tag_to_id=tag_to_id,
        lower=args.lower,
        batch_size=args.batch_size,
        sampling_ratio=args.training_ratio,
        use_ori_text_char=args.use_ori_text_char,
        ds_gen_mode=args.training_gen_mode,
        singletons=singletons,
        singleton_unk_probability=args.singleton_unk_probability,
        bert_tokenizer=bert_tokenizer,
    )
    val_X, val_Y, val_data, val_sentences = loader.prepare_dataset(
        sentences=val_sentences,
        word_to_id=word_to_id,
        char_to_id=char_to_id,
        tag_to_id=tag_to_id,
        lower=args.lower,
        batch_size=args.batch_size,
        use_ori_text_char=args.use_ori_text_char,
        bert_tokenizer=bert_tokenizer,
    )
    test_X, test_Y, test_data, test_sentences = loader.prepare_dataset(
        sentences=test_sentences,
        word_to_id=word_to_id,
        char_to_id=char_to_id,
        tag_to_id=tag_to_id,
        lower=args.lower,
        batch_size=args.batch_size,
        use_ori_text_char=args.use_ori_text_char,
        bert_tokenizer=bert_tokenizer,
    )

    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(val_data), len(test_data)))

    if args.training_gen_mode == 'static':
        args.steps_per_epoch = len(list(train_X))


    #########################################
    # build model
    #########################################
    if not args.to_reload_model:
        # Initialize model
        print('Initializing model...')
        model = model_framework.NERModel(
            model_path=args.model_dir,
            model_name='test_model',
            to_reload_model=False,
            tag_scheme=args.tag_scheme,
            clean_tag=args.clean_tag,
            id_to_word=id_to_word,
            id_to_char=id_to_char,
            id_to_tag=id_to_tag,
            pre_embedding=pre_emb,
            classifier_type=args.classifier_type,
            bert_path=args.bert_path,
            bert_first_trainable_layer=args.bert_first_trainable_layer,
            word_dim=args.word_dim,
            word_lstm_dim=args.word_lstm_dim,
            word_bidirect=args.word_bidirect,
            word_unroll=args.word_unroll,
            word_rnn_wrapper=args.word_rnn_wrapper,
            char_dim=args.char_dim,
            char_lstm_dim=args.char_lstm_dim,
            char_bidirect=args.char_bidirect,
            char_combine_method=args.char_combine_method,
            char_unroll=args.char_unroll,
            char_rnn_wrapper=args.char_rnn_wrapper,
            ele_num=args.use_ele_num,
            only_CHO=args.use_only_CHO,
            tar_tag=args.test_fake_tar_tag,
            pre_tag=args.test_fake_pre_tag,
            rnn_type=args.rnn_type,
            lower=args.lower,
            zeros=args.zeros,
            use_ori_text_char=args.use_ori_text_char,
            crf=args.crf,
            crf_begin_end=args.crf_begin_end,
            dropout=args.dropout,
            lr_method=args.lr_method,
            loss_per_token=args.loss_per_token,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            steps_per_epoch=args.steps_per_epoch,
        )
        print("Model location: %s" % model.model_path)
    else:
        # Reload previous model values
        print('Reloading previous model...')
        model = model_framework.NERModel.reload_model(
            model_path=args.model_dir,
            bert_path=args.bert_path,
        )

    #########################################
    # callbacks
    #########################################
    validation_callback = callbacks.GetScoreCallback(
        data_x=val_X,
        data_y=val_X.map(lambda x: x['tags']),
        data_raw_sentences=val_sentences,
    )

    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=model.last_cp_path,
        save_weights_only=True,
        verbose=0
    )

    #########################################
    # train model
    #########################################
    # TODO: test if prefetch dataset could accelerate it
    train_hist = model.fit(
        x=tf.data.Dataset.zip((train_X, train_Y)),
        epochs=args.num_epochs,
        steps_per_epoch=None \
            if args.training_gen_mode == 'static' \
            else args.steps_per_epoch,
        callbacks=[
            validation_callback,
            cp_callback,
        ],
        verbose=0,
    )

    print('model.summary()', model.summary())

    #########################################
    # test
    #########################################
    print('-------------test------------------')
    model_opt = model_framework.NERModel.reload_model(
        model_path=args.model_dir,
        bert_path=args.bert_path,
    )
    val_score = model_opt.evaluate_id(
        x_batches=val_X,
        y_true=val_X.map(lambda x: x['tags']),
        raw_sentences=val_sentences,
    )
    print('val score', val_score)
    test_score = model_opt.evaluate_id(
        x_batches=test_X,
        y_true=test_X.map(lambda x: x['tags']),
        raw_sentences=test_sentences,
    )
    print('test score', test_score)

    print('end time', datetime.datetime.now())

