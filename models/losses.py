from torch.nn.modules.loss import *


loss_dict = {"MSE": MSELoss,
             "BCE": BCELoss,
             "CE": CrossEntropyLoss,
             "BCEL": BCEWithLogitsLoss,
             "MLML": MultiLabelMarginLoss}
