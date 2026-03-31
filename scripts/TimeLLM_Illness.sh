model_name=TimeLLM
train_epochs=50
learning_rate=0.01
llama_layers=16

master_port=29597
num_process=8
batch_size=16
eval_batch_size=4
d_model=16
d_ff=32

comment='TimeLLM-Illness'

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id illness_96_24 \
  --model $model_name \
  --data Illness \
  --features M \
  --freq W \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 8 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --eval_batch_size $eval_batch_size \
  --use_amp \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --loss MSE \
  --num_tokens 100 \
  --model_comment $comment

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id illness_96_36 \
  --model $model_name \
  --data Illness \
  --features M \
  --freq W \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 36 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 8 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --eval_batch_size $eval_batch_size \
  --use_amp \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --loss MSE \
  --num_tokens 100 \
  --model_comment $comment

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id illness_96_48 \
  --model $model_name \
  --data Illness \
  --features M \
  --freq W \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 8 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --eval_batch_size $eval_batch_size \
  --use_amp \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --loss MSE \
  --num_tokens 100 \
  --model_comment $comment

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id illness_96_60 \
  --model $model_name \
  --data Illness \
  --features M \
  --freq W \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 60 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 8 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --eval_batch_size $eval_batch_size \
  --use_amp \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --loss MSE \
  --num_tokens 100 \
  --model_comment $comment
