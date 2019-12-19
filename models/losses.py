from torch.nn.modules.loss import *


loss_dict = {"MSE": MSELoss,
             "BCE": BCELoss,
             "BCEL": BCEWithLogitsLoss,
             "MLML": MultiLabelMarginLoss}
