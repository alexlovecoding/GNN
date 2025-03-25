import optuna
import os
import time
from datetime import datetime  # 用于获取时间戳

from grnn_new import get_args, train_model, get_data

# 定义数据库文件来保存优化进度
db_path = 'sqlite:///optuna_search.db'
n_trials = 100  # 设置每轮搜索的试验次数

# 定义目标函数
def objective(trial):
    # 定义要优化的参数范围
    args = get_args()  # 获取默认参数
    # args.learning_rate = trial.suggest_float('learning_rate', 5e-4, 1e-3, log=True)
    # args.batch_size = trial.suggest_categorical('batch_size', [16, 32])
    # args.latent_dim1 = trial.suggest_int('latent_dim1', 100, 130)
    # args.latent_dim2 = trial.suggest_int('latent_dim2', 8, 10)
    # args.latent_dim3 = trial.suggest_int('latent_dim3', 2, 5)
    # args.lstm_dropout = trial.suggest_float('lstm_dropout', 0.5, 0.6)
    # args.weight_decay = trial.suggest_float('weight_decay', 2e-4, 3e-4, log=True)
    # args.lr_decay_rate = trial.suggest_float('lr_decay_rate', 0.85, 0.9)
    # args.hid_size = trial.suggest_int('hid_size', 5, 15)
    # args.num_layers = trial.suggest_int('num_layers', 1, 3)
    args.manualseed = trial.suggest_int('num_layers', 1, 10000)

    # # 输出所有参数
    # print(f"Trial parameters: \n"
    #       f"learning_rate={args.learning_rate}, batch_size={args.batch_size}, \n"
    #       f"latent_dim1={args.latent_dim1}, latent_dim2={args.latent_dim2}, latent_dim3={args.latent_dim3}, \n"
    #       f"lstm_dropout={args.lstm_dropout}, weight_decay={args.weight_decay}, \n"
    #       f"lr_decay_rate={args.lr_decay_rate}, hid_size={args.hid_size}, num_layers={args.num_layers}")

    # 加载数据集
    train_dataloader, validation_dataloader, test_dataloader = get_data(args)

    # 训练模型并获取验证集的损失 (validation_mse)
    validation_mse = train_model(args, train_dataloader, validation_dataloader, test_dataloader)

    return validation_mse

# 如果数据库文件已存在，则加载已存在的study，否则创建新的study
if os.path.exists("optuna_search.db"):
    study = optuna.load_study(study_name="gnn_rnn_study", storage=db_path)

    # 检查是否已经完成
    if study.user_attrs.get('is_completed', False):
        print("上一次搜索已完成，创建新的搜索")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"gnn_rnn_study_{timestamp}"
        study = optuna.create_study(study_name=study_name, direction='minimize', storage=db_path)
        study.set_user_attr('n_trials', n_trials)  # 保存新轮搜索的n_trials数
        study.set_user_attr('is_completed', False)  # 标记为未完成
    else:
        # 如果未完成，继续之前的搜索
        print("继续上一次未完成的搜索")
else:
    study = optuna.create_study(study_name="gnn_rnn_study", direction='minimize', storage=db_path)
    study.set_user_attr('n_trials', n_trials)  # 保存搜索轮次
    study.set_user_attr('is_completed', False)  # 标记为未完成

# 记录开始时间
start_time = time.time()

# 优化搜索
study.optimize(objective, n_trials=n_trials)
# study.optimize(objective, n_trials=n_trials, n_jobs=2)

# 记录结束时间
end_time = time.time()

# 计算总耗时
total_time = end_time - start_time
total_time_minutes = total_time / 60  # 转换为分钟
total_time_hours = total_time_minutes / 60  # 转换为小时

# 搜索完成后，更新标记为已完成
study.set_user_attr('is_completed', True)

# 输出最佳参数
print('最佳参数:', study.best_params)
print('最优验证集MSE:', study.best_value)
print(f"总搜索时间: {total_time:.2f} 秒 ({total_time_minutes:.2f} 分钟, {total_time_hours:.2f} 小时)")

# 获取当前时间戳
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 创建文件名，格式为 "时间戳+轮数.txt"
file_name = f'{timestamp}_{n_trials}_trials.txt'

# 将最佳参数组合保存到文件
with open(file_name, 'w') as f:
    f.write(f"Best parameters (from {n_trials} trials):\n")
    for key, value in study.best_params.items():
        f.write(f"{key}: {value}\n")
    f.write(f"\nBest Validation MSE: {study.best_value}\n")
    f.write(f"\nTotal search time: {total_time:.2f} seconds ({total_time_minutes:.2f} minutes, {total_time_hours:.2f} hours)\n")

print(f"最佳参数已保存到文件: {file_name}")
