# Deep-learning-for-contact\_map\_v2

Deep learning method for prediction of pretein contact map, with predicted contact map by other software(for example CCMpred, PSICOV, and so on) as input.

## Requirements

- [python2.7]()
- [Tensorflow (>= 1.0.0)](https://www.tensorflow.org/install/install_linux)
- [Numpy](https://github.com/numpy/numpy/blob/master/INSTALL.rst.txt)

## Need to do
1. Get protein structure 1D features(for example sequence, sse, ACA, and so on), and 2D features(for example predicted CCMpred, PSICOV and other pairwise feature)
2. Modify `read_into_tfrecord.py`, and used it to transfer your data to tfrecord
3. set your own config in `libs/config/config.py`
4. run `train.py`
