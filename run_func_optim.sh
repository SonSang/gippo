ITER=5

# Dejong
for (( i=1; i<=${ITER}; i++ ))
do
    python ./train.py --cfg ./config/func_optim/dejong/lr.yaml --logdir ./logdir/func_optim/dejong/lr/ --seed ${i} --device cpu
    python ./train.py --cfg ./config/func_optim/dejong/rp.yaml --logdir ./logdir/func_optim/dejong/rp/ --seed ${i} --device cpu
    python ./train.py --cfg ./config/func_optim/dejong/lrp.yaml --logdir ./logdir/func_optim/dejong/lrp/ --seed ${i} --device cpu
    python ./train.py --cfg ./config/func_optim/dejong/ppo.yaml --logdir ./logdir/func_optim/dejong/ppo/ --seed ${i} --device cpu
    python ./train.py --cfg ./config/func_optim/dejong/gippo.yaml --logdir ./logdir/func_optim/dejong/gippo/ --seed ${i} --device cpu
done

# Dejong 64
for (( i=1; i<=${ITER}; i++ ))
do
    python ./train.py --cfg ./config/func_optim/dejong64/lr.yaml --logdir ./logdir/func_optim/dejong64/lr/ --seed ${i} --device cpu
    python ./train.py --cfg ./config/func_optim/dejong64/rp.yaml --logdir ./logdir/func_optim/dejong64/rp/ --seed ${i} --device cpu
    python ./train.py --cfg ./config/func_optim/dejong64/lrp.yaml --logdir ./logdir/func_optim/dejong64/lrp/ --seed ${i} --device cpu
    python ./train.py --cfg ./config/func_optim/dejong64/ppo.yaml --logdir ./logdir/func_optim/dejong64/ppo/ --seed ${i} --device cpu
    python ./train.py --cfg ./config/func_optim/dejong64/gippo.yaml --logdir ./logdir/func_optim/dejong64/gippo/ --seed ${i} --device cpu
done

# Ackley
for (( i=1; i<=${ITER}; i++ ))
do
    python ./train.py --cfg ./config/func_optim/ackley/lr.yaml --logdir ./logdir/func_optim/ackley/lr/ --seed ${i} --device cpu
    python ./train.py --cfg ./config/func_optim/ackley/rp.yaml --logdir ./logdir/func_optim/ackley/rp/ --seed ${i} --device cpu
    python ./train.py --cfg ./config/func_optim/ackley/lrp.yaml --logdir ./logdir/func_optim/ackley/lrp/ --seed ${i} --device cpu
    python ./train.py --cfg ./config/func_optim/ackley/ppo.yaml --logdir ./logdir/func_optim/ackley/ppo/ --seed ${i} --device cpu
    python ./train.py --cfg ./config/func_optim/ackley/gippo.yaml --logdir ./logdir/func_optim/ackley/gippo/ --seed ${i} --device cpu
done

# Ackley 64
for (( i=1; i<=${ITER}; i++ ))
do
    python ./train.py --cfg ./config/func_optim/ackley64/lr.yaml --logdir ./logdir/func_optim/ackley64/lr/ --seed ${i} --device cpu
    python ./train.py --cfg ./config/func_optim/ackley64/rp.yaml --logdir ./logdir/func_optim/ackley64/rp/ --seed ${i} --device cpu
    python ./train.py --cfg ./config/func_optim/ackley64/lrp.yaml --logdir ./logdir/func_optim/ackley64/lrp/ --seed ${i} --device cpu
    python ./train.py --cfg ./config/func_optim/ackley64/ppo.yaml --logdir ./logdir/func_optim/ackley64/ppo/ --seed ${i} --device cpu
    python ./train.py --cfg ./config/func_optim/ackley64/gippo.yaml --logdir ./logdir/func_optim/ackley64/gippo/ --seed ${i} --device cpu
done