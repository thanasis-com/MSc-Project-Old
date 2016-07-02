
for WD in 0
do
	THEANO_FLAGS=mode=FAST_RUN,device=$2,floatX=float32 python main.py $1 $WD 1000 >> $3
done
