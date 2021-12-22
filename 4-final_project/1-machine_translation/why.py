from model.transformer import Transformer, ScheduledOptim
from torch.optim import Adam
from matplotlib import pyplot as plt
import numpy as np

model = Transformer()


optimizer = ScheduledOptim(Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09), lr_mul=2.0, d_model=512, n_warmup_steps=4000)
lr_list = []
for step in range(9000):
    optimizer.step_and_update_lr()
    lr = optimizer._optimizer.param_groups[0]['lr']
    lr_list.append(lr)
    # print('Step: {}|    lr: {}'.format(step, lr))

plt.plot(np.array(lr_list))
plt.show()
