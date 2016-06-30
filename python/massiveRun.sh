
for LR in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.2
do
	THEANO_FLAGS=mode=FAST_RUN,device=$2,floatX=float32 python main.py $LR $1 1000 >> res.txt
done
