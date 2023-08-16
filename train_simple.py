import itertools

import torch
import torch.optim as optim
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from realworldnlp.seq2multiseq_reader import Seq2MultiSeqDatasetReader

from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.activations import Activation
# from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.attention import LinearAttention, BilinearAttention, DotProductAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SimpleSeq2SeqPredictor
from allennlp.common.params import Params
from allennlp.commands.train import create_serialization_dir
#from allennlp.training.trainer import Trainer
from realworldnlp.custom_trainer import CustomTrainer as Trainer

from realworldnlp.simple_seq2seq import SimpleSeq2Seq
from realworldnlp.simple_vae import SimpleVAE
from realworldnlp.spsved import SPSVED
from realworldnlp.attention_encoder import AttentionEncoder

import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="quora",
    type=str,
)
parser.add_argument(
    "--max_kl_weight",
    default=1e-3,
    type=float,
)
parser.add_argument(
    "--toy_test",
    default=False,
    type=str2bool,
    help="toy_test",
)
parser.add_argument(
    "--epochs",
    default=10,
    type=int,
    help="epochs",
)
parser.add_argument(
    "--recover",
    default=False,
    type=str2bool,
    help="recover",
)

parser.add_argument(
    "--model",
    default="simple_vae",
    type=str,
    help="model name",
)
parser.add_argument(
    "--mincount",
    default=5,
    type=int,
    help="min vocab count",
)

parser.add_argument(
    "--newtargetemd",
    default=False,
    type=str2bool,
    help="different target embedding from the encoder",
)

parser.add_argument(
    "--negative_penalize",
    default=False,
    type=str2bool,
    help="negative_penalize",
)

parser.add_argument(
    "--use_attn",
    default=False,
    type=str2bool,
    help="use_attn",
)

parser.add_argument(
    "--use_cell",
    default=False,
    type=str2bool,
    help="use_cell",
)

parser.add_argument(
    "--is_adv",
    default=False,
    type=str2bool,
    help="is_adv",
)

parser.add_argument(
    "--is_adv_x",
    default=False,
    type=str2bool,
    help="is_adv",
)



parser.add_argument(
    "--lr",
    default=1e-3,
    type=float,
)
parser.add_argument(
    "--is_test",
    default=False,
    type=str2bool,
    help="recover",
)
parser.add_argument(
    "--is_wsc",
    default=False,
    type=str2bool,
    help="is_wsc",
)
parser.add_argument(
    "--is_ssc",
    default=False,
    type=str2bool,
    help="is_wsc",
)

parser.add_argument(
    "--pretrain",
    default=False,
    type=str2bool,
    help="pretrain",
)





adv_beta = 0.5
adv_beta_x = 0.5

wsc_beta = 0.5
ssc_beta = 0.5



args = parser.parse_args()

print(args)


is_adv = args.is_adv
is_adv_x = args.is_adv_x
is_disentangle = is_adv

is_wsc = args.is_wsc
is_ssc = args.is_ssc

use_glove = True
use_attn = args.use_attn
use_cell = args.use_cell

keep_all_vocab = True

negative_penalize = args.negative_penalize

dataset_name = args.dataset #'quora'#'mscoco'
model = args.model #"simple_vae" # seq2seq, simple_vae

max_kl_weight = args.max_kl_weight

if (model=="simple_vae"):
    assert(use_attn == False)

use_multi_ref = True
CurrentDatasetReader = Seq2MultiSeqDatasetReader

min_count = args.mincount
guide_metric = '+CustomBLEU'
if (dataset_name == 'mscoco'):
    max_decoding_steps = 16
else:
    max_decoding_steps = 20


if (use_glove):
    EN_EMBEDDING_DIM = 300
else:
    EN_EMBEDDING_DIM = 500
    

serialization_dir = "./dir_"+dataset_name+"_" + model
lr = args.lr
if (lr != 1e-3):
    serialization_dir += ("_lr"+str(lr))

if (model=="simple_vae"):
    serialization_dir += ("_klweight_" + str(max_kl_weight))

if (use_attn):
    serialization_dir += "_attn"
if (keep_all_vocab):
    serialization_dir += "_keepvocab"
if (negative_penalize):
    serialization_dir += "_negapenalize"
if (is_adv):
    serialization_dir += "_adv"
if (is_adv_x):
    assert (is_adv == True)
    serialization_dir += "_adx"
if (is_wsc):
    serialization_dir += "_wsc"
if (is_ssc):
    serialization_dir += "_ssc"
    
    
ZH_EMBEDDING_DIM = EN_EMBEDDING_DIM
HIDDEN_DIM = 500
CUDA_DEVICE = 0

pretrain_emd_file = "../../../../glove/glove.6B.300d.txt"


recover = args.recover
toy_test = args.toy_test
n_epoch = args.epochs

params = Params({})
if (toy_test):
    serialization_dir += "_toytest"
if (not recover):
    print("creating dir", serialization_dir)
    create_serialization_dir(params, serialization_dir, recover=False, force=True)
else:
    print("recovering from dir", serialization_dir)
    
    
tokenizer = WordTokenizer()
token_indexer = SingleIdTokenIndexer(lowercase_tokens=True)



    
reader = CurrentDatasetReader(
    source_tokenizer=tokenizer,
    target_tokenizer=tokenizer,
    source_token_indexers={'tokens': token_indexer},
    target_token_indexers={'tokens': token_indexer})


if (toy_test):
    validation_dataset = reader.read('data/'+dataset_name+'/paraphrases.dev.tsv')
    train_dataset = validation_dataset
    test_dataset = validation_dataset
else:
    train_dataset = reader.read('data/'+dataset_name+'/paraphrases.train.tsv')
    validation_dataset = reader.read('data/'+dataset_name+'/paraphrases.dev.tsv')
    test_dataset = reader.read('data/'+dataset_name+'/paraphrases.test.tsv')

if (keep_all_vocab):
    vocab = Vocabulary.from_instances(train_dataset + validation_dataset + test_dataset,
                                      min_count={'tokens': min_count})
else:
    vocab = Vocabulary.from_instances(train_dataset + validation_dataset,
                                      min_count={'tokens': min_count})

print(vocab)


if (use_glove):
    en_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                         embedding_dim=EN_EMBEDDING_DIM,
                         pretrained_file=pretrain_emd_file,
                         trainable=True)
    if (args.newtargetemd):
        target_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             embedding_dim=EN_EMBEDDING_DIM,
                             pretrained_file=pretrain_emd_file,
                             trainable=True)
    else:
        target_embedding = en_embedding
else:
    en_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                         embedding_dim=EN_EMBEDDING_DIM)
    if (args.newtargetemd):
        target_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             embedding_dim=EN_EMBEDDING_DIM)
    else:
        target_embedding = en_embedding
    
if (model != "spsved"):
    assert(is_adv == False)
    encoder = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(EN_EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True, num_layers=2))
    
#encoder = StackedSelfAttentionEncoder(input_dim=EN_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, projection_dim=128, feedforward_hidden_dim=128, num_layers=1, num_attention_heads=8)

source_embedder = BasicTextFieldEmbedder({"tokens": en_embedding})
target_embedder = BasicTextFieldEmbedder({"tokens": target_embedding})

# attention = LinearAttention(HIDDEN_DIM, HIDDEN_DIM, activation=Activation.by_name('tanh')())
# attention = BilinearAttention(HIDDEN_DIM, HIDDEN_DIM)


if (use_attn):
    attention = DotProductAttention()
else:
    attention = None
    
latent_disc = None

disc_optimizer = None
disc_model = None

if (model == "seq2seq"):
     model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                      target_embedder = target_embedder.token_embedder_tokens,
                      target_embedding_dim=ZH_EMBEDDING_DIM,
                      target_namespace='tokens',
                      attention=attention,
                      beam_size=1,
                      use_bleu=True,
                      use_rouge=False,
                      use_bleu2=False,
                      scheduled_sampling_ratio=0.2,
                      multi_ref=use_multi_ref,
                      save_dir=serialization_dir)
elif (model == "simple_vae"):
    model = SimpleVAE(vocab, source_embedder, encoder, max_decoding_steps,
                      target_embedder = target_embedder.token_embedder_tokens,
                      target_embedding_dim=ZH_EMBEDDING_DIM,
                      target_namespace='tokens',
                      attention=attention,
                      beam_size=1,
                      use_bleu=True,
                      use_rouge=False,
                      use_bleu2=False,
                      scheduled_sampling_ratio=0.2,
                      multi_ref=use_multi_ref,
                      save_dir=serialization_dir)
elif (model == "dss_vae"):
    
    semantic_encoder = PytorchSeq2SeqWrapper(
    torch.nn.LSTM(EN_EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True, num_layers=2))
    
    syntax_z_dim = 500
    semantic_z_dim = 500
    syn2sem = torch.nn.Linear(syntax_z_dim, semantic_z_dim)
    latent_disc = torch.nn.Linear(semantic_z_dim * 3, 2)
    
    from itertools import chain
    
    latent_disc_params = chain(
        syn2sem.parameters(), latent_disc.parameters()
    )
    disc_model = torch.nn.ModuleList([syn2sem, latent_disc])
    disc_model.cuda()
    
    disc_optimizer = optim.Adam(disc_model.parameters(), lr=lr)
    
    is_adv = True
    
    model = SimpleVAE(vocab, source_embedder, encoder, max_decoding_steps,
                      target_embedder = target_embedder.token_embedder_tokens,
                      target_embedding_dim=ZH_EMBEDDING_DIM,
                      target_namespace='tokens',
                      attention=attention,
                      beam_size=1,
                      use_bleu=True,
                      use_rouge=False,
                      use_bleu2=False,
                      scheduled_sampling_ratio=0.2,
                      multi_ref=use_multi_ref,
                      save_dir=serialization_dir,
                      is_disentangle=True,
                      semantic_encoder=semantic_encoder,
                      syntax_z_dim = syntax_z_dim,
                      semantic_z_dim = semantic_z_dim,
                      adv_beta=adv_beta,
                      negative_penalize=negative_penalize)
    
elif (model == "spsved"):
    
    ATTN_ENCODE_DIM = HIDDEN_DIM
    
    target_encoder = PytorchSeq2SeqWrapper(
    torch.nn.LSTM(EN_EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True, num_layers=2))
    
    semantic_encoder = PytorchSeq2SeqWrapper(
    torch.nn.LSTM(EN_EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True, num_layers=2))
    
    z_dim = 500
    semantic_disc_dim = 500

    latent_disc = torch.nn.Linear(semantic_disc_dim *2 * 4, 2)
    
    from itertools import chain
    
    if (is_adv_x):
        x_recontor_decoder = PytorchSeq2SeqWrapper(
    torch.nn.LSTM(EN_EMBEDDING_DIM + target_encoder.get_output_dim(), HIDDEN_DIM, batch_first=True, bidirectional=False, num_layers=1))
        x_recontor_fc = torch.nn.Linear(HIDDEN_DIM, vocab.get_vocab_size())
        x_recontor = torch.nn.ModuleList([x_recontor_decoder, x_recontor_fc])
        
        disc_model = torch.nn.ModuleList([x_recontor, latent_disc])
    else:
        disc_model = latent_disc
        
    disc_model.cuda()
    
    disc_optimizer = optim.Adam(disc_model.parameters(), lr=lr)
    
    is_adv = is_adv
    
    model = SPSVED(vocab, source_embedder, max_decoding_steps,
                      target_embedder = target_embedder.token_embedder_tokens,
                      target_embedding_dim=ZH_EMBEDDING_DIM,
                      target_namespace='tokens',
                      attention=attention,
                      beam_size=1,
                      use_bleu=True,
                      use_rouge=False,
                      use_bleu2=False,
                      use_cell = use_cell,
                      scheduled_sampling_ratio=0.2,
                      multi_ref=use_multi_ref,
                      save_dir=serialization_dir,
                      is_disentangle=is_disentangle,
                      target_encoder = target_encoder,
                      semantic_encoder=semantic_encoder,
                      z_dim = z_dim,
                      adv_beta=adv_beta,
                      negative_penalize=negative_penalize,
                      max_kl_weight=max_kl_weight,
                      is_adv_x = is_adv_x,
                      adv_beta_x=adv_beta_x,
                      is_wsc=is_wsc,
                      is_ssc=is_ssc,
                      wsc_beta=wsc_beta)
    



model.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)

iterator = BucketIterator(batch_size=32, sorting_keys=[("source_tokens", "num_tokens")])

iterator.index_with(vocab)

trainer = Trainer(is_adv=is_adv,
                  disc_model=disc_model,
                  disc_optimizer = disc_optimizer,
                  model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  num_epochs=args.epochs,
                  patience=8,
                  num_serialized_models_to_keep=1,
                  validation_metric=guide_metric,
                  serialization_dir=serialization_dir,
                  cuda_device=CUDA_DEVICE)


trainer.sample_predictor = SimpleSeq2SeqPredictor(model, reader)

model.kl_weight = 0
model.max_kl_weight = max_kl_weight


from allennlp.training.util import evaluate
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import BLEU

if (args.pretrain):
    
    if (True):
        model._is_wsc = False
        model._is_ssc = False

        trainer.is_adv = True
        model._is_disentangle = True

        print("pretraining...")
        metrics = trainer.train()
        best_model_state_path = os.path.join(serialization_dir, "best.th")
        pretrain_state_path = os.path.join(serialization_dir, "pretrain.th")
        os.popen('cp '+best_model_state_path+' '+pretrain_state_path)


        best_model_state = torch.load(best_model_state_path)
        model.load_state_dict(state_dict=best_model_state)

        #-------------
        test_metric = evaluate(model,
                 instances = test_dataset,
                 data_iterator = iterator,
                 cuda_device = CUDA_DEVICE,
                 batch_weight_key = None)

        model._id2str.load_saved_sentences()
        rouge_metric = model._id2str.get_rouge_scores()

        from sentence_transformers import SentenceTransformer
        semantic_model = SentenceTransformer('bert-base-nli-mean-tokens')
        semantic_model.cuda()
        print("move semantic_model to gpu")

        bertcs_metric = model._id2str.get_BERTCS(semantic_model)

        test_metric.update(rouge_metric)
        test_metric.update(bertcs_metric)

        print(test_metric)

        import json
        saved_metric_file = os.path.join(serialization_dir, "pretrain_test_metric.json")
        with open(saved_metric_file, "w") as f:
            json.dump(str(test_metric), f)
        print("save test_metric to ", saved_metric_file)
        #-------------
    else:

        best_model_state_path = os.path.join(serialization_dir, "best.th")
        pretrain_state_path = os.path.join(serialization_dir, "pretrain.th")

        best_model_state = torch.load(best_model_state_path)
        model.load_state_dict(state_dict=best_model_state)


    for group in trainer.optimizer.param_groups:
        group['lr'] = 1e-4
    for group in trainer.disc_optimizer.param_groups:
        group['lr'] = 1e-4
    
    print("after pretrain, training")
    model._is_disentangle = is_adv
    model._is_wsc = is_wsc
    model._is_ssc = is_ssc
    trainer.is_adv = is_adv
    trainer.train(recover=False)
    
else:
    if (not args.is_test):
        metrics = trainer.train()
# if (recover):
#     metrics = trainer.train(recover=False)





best_model_state_path = os.path.join(serialization_dir, "best.th")
best_model_state = torch.load(best_model_state_path)
best_model = model
best_model.load_state_dict(state_dict=best_model_state)


test_metric = evaluate(model,
             instances = test_dataset,
             data_iterator = iterator,
             cuda_device = CUDA_DEVICE,
             batch_weight_key = None)

model._id2str.load_saved_sentences()
rouge_metric = model._id2str.get_rouge_scores()

if (not args.pretrain):
    from sentence_transformers import SentenceTransformer
    semantic_model = SentenceTransformer('bert-base-nli-mean-tokens')
    semantic_model.cuda()
    print("move semantic_model to gpu")

bertcs_metric = model._id2str.get_BERTCS(semantic_model)

test_metric.update(rouge_metric)
test_metric.update(bertcs_metric)
    
print(test_metric)

import json
saved_metric_file = os.path.join(serialization_dir, "test_metric.json")
with open(saved_metric_file, "w") as f:
    json.dump(str(test_metric), f)
print("save test_metric to ", saved_metric_file)