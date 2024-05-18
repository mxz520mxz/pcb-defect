cur_dir=`dirname $0`
cp $cur_dir/act_*.sh $CONDA_PREFIX/etc/conda/activate.d/
cp $cur_dir/deact_*.sh $CONDA_PREFIX/etc/conda/deactivate.d/
