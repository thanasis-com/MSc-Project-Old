
for LR in 0.0001
do
	THEANO_FLAGS=mode=FAST_RUN,device=$2,floatX=float32 python main.py $LR $1 1 >> res.txt
done
