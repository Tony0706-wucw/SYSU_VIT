import wandb

# 启动离线模式
wandb.init(project="test-project", entity="your_username", mode="offline")

# 记录一些数据
wandb.log({"accuracy": 0.9})

# 结束 run
wandb.finish()