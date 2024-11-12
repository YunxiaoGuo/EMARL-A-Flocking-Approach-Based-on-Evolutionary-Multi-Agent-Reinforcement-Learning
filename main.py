import os
from trainer import training
from parameters.parameters import get_paras
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"




if __name__ == '__main__':
    args = get_paras()
    training(args)
