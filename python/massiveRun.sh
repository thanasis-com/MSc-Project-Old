
for WD in 0.0001
do
	THEANO_FLAGS=mode=FAST_RUN,device=$2,floatX=float32 python main.py $1 $WD 1 >> $3
done
