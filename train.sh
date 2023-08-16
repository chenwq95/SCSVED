# python -u train_simple.py --dataset quora --max_kl_weight 1e-3 --toy_test False --epochs 15 --newtargetemd True --mincount 1
# python -u train_simple.py --dataset quora --max_kl_weight 1e-4 --toy_test False --epochs 15
# python -u train_simple.py --dataset mscoco --max_kl_weight 1e-3 --toy_test False --epochs 15
# python -u train_simple.py --dataset mscoco --max_kl_weight 1e-4 --toy_test False --epochs 15
# python -u train_simple.py --dataset quora --max_kl_weight 1e-2 --toy_test False --epochs 15
# python -u train_simple.py --dataset quora --max_kl_weight 1e-1 --toy_test False --epochs 15
# python -u train_simple.py --dataset quora --max_kl_weight 1 --toy_test False --epochs 15


# python -u train_simple.py --dataset quora --model spsved --max_kl_weight 100 --toy_test True --epochs 1 --lr 5e-4 --mincount 1 --use_cell True --use_attn True --is_adv True --is_wsc True --pretrain True 


# python -u train_simple.py --dataset quora --model spsved --max_kl_weight 100 --toy_test False --epochs 15 --lr 5e-4 --mincount 1 --use_cell True --use_attn True --is_adv True --is_wsc True --pretrain True 

# python -u train_simple.py --dataset mscoco --model spsved --max_kl_weight 100 --toy_test False --epochs 10 --lr 5e-4 --mincount 1 --use_cell True --use_attn True --is_adv True --is_wsc True --pretrain True 


# python -u train_simple.py --dataset quora --model spsved --max_kl_weight 500 --toy_test True --epochs 1 --lr 1e-3 --mincount 1 --use_cell True --use_attn True --is_adv True --is_wsc True --is_ssc True --pretrain True --recover True


# python -u train_simple.py --dataset quora --model spsved --max_kl_weight 500 --toy_test False --epochs 15 --lr 1e-3 --mincount 1 --use_cell True --use_attn True --is_adv True --is_wsc True --is_ssc True --pretrain True --recover True

python -u train_simple.py --dataset quora --model spsved --max_kl_weight 100 --toy_test False --epochs 15 --lr 5e-4 --mincount 1 --use_cell True --use_attn True --is_adv True --is_wsc True --is_ssc True --pretrain True --recover True


python -u train_simple.py --dataset mscoco --model spsved --max_kl_weight 100 --toy_test False --epochs 15 --lr 5e-4 --mincount 1 --use_cell True --use_attn True --is_adv True --is_wsc True --is_ssc True --pretrain True --recover True