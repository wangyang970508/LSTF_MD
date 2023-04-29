model_name=LSTF_MD

for pred_len in 96 192 336 720
do
for seq_len in 96
do
   python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id Electricity_$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len  \
    --enc_in 321 \
    --des 'Exp' \
    --itr 1 --batch_size 16  --learning_rate 0.001 >logs/sh/$model_name'_'electricity_$seq_len'_'$pred_len.log
    

      python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id Electricity_$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features S \
    --seq_len $seq_len \
    --pred_len $pred_len  \
    --enc_in 321 \
    --des 'Exp' \
    --itr 1 --batch_size 16  --learning_rate 0.001 >logs/sh/$model_name'_'electricity_$seq_len'_'$pred_len'_'S.log

done
done