import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def read_test_output(file_path):
    predictions = []
    true_values = []

    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()[1:]]  # 跳过首行并去除空白

    i = 0
    while i < len(lines):
        if lines[i].startswith('sample-'):
            try:
                # 检查是否有足够的行
                if i + 5 >= len(lines):
                    print(f"样本 {lines[i]} 数据不完整，跳过")
                    i += 1
                    continue

                # 解析预测值 (i+3行)
                pred_line = lines[i + 3]
                try:
                    pred_values = list(
                        map(lambda x: float(x.strip()), filter(lambda x: x.strip(), pred_line.split(','))))
                except ValueError:
                    print(f"无效预测数据在样本 {lines[i]} 行 {i + 3}: {pred_line}")
                    i += 7
                    continue

                # 解析真实值 (i+5行)
                true_line = lines[i + 5]
                try:
                    true_values_sample = list(
                        map(lambda x: float(x.strip()), filter(lambda x: x.strip(), true_line.split(','))))
                except ValueError:
                    print(f"无效真实数据在样本 {lines[i]} 行 {i + 5}: {true_line}")
                    i += 7
                    continue

                predictions.append(pred_values)
                true_values.append(true_values_sample)
                i += 7  # 跳到下一个sample起始位置
            except (IndexError, ValueError) as e:
                print(f"解析错误在样本 {lines[i]} (行 {i}): {str(e)}")
                i += 1  # 逐步排查错误
        else:
            i += 1

    return np.array(predictions), np.array(true_values)


def plot_mret_distribution(predictions, true_values):
    mret = np.abs(predictions - true_values) / np.abs(true_values) * 100
    mret_values = mret.flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(mret_values, bins=20, density=True, color='skyblue', edgecolor='black')
    plt.title('测试数据集的 MRET 分布')
    plt.xlabel('MRET (%)')
    plt.ylabel('概率密度')
    plt.grid(linestyle='--', alpha=0.7)

    plt.savefig('mret_distribution.png')
    plt.show()


def plot_comparison_at_epsilon(predictions, true_values, epsilon_index=9):
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    pred_at_epsilon = predictions[:, epsilon_index]
    true_at_epsilon = true_values[:, epsilon_index]

    plt.figure(figsize=(8, 8))
    plt.scatter(true_at_epsilon, pred_at_epsilon, c='#FF6B6B', edgecolors='k', alpha=0.8, label='预测值')

    # 绘制理想预测线 y = x
    min_val = min(np.min(true_at_epsilon), np.min(pred_at_epsilon))
    max_val = max(np.max(true_at_epsilon), np.max(pred_at_epsilon))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='理想预测线 (y = x)')

    # 绘制 y = x + 1.5 线
    plt.plot([min_val, max_val], [min_val + 0.5, max_val + 0.5], 'b--', lw=2, label='y = x + 1.5')

    # 绘制 x = y + 1.5 线
    plt.plot([min_val + 0.5, max_val + 0.5], [min_val, max_val], 'g--', lw=2, label='x = y + 1.5')

    plt.title('GNN 预测的 $f/f_0$ 与 CPFEM 仿真值对比 ($\\epsilon_a=10\\%$)')
    plt.xlabel('CPFEM 仿真值')
    plt.ylabel('GNN 预测值')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('45deg_line.png')
    plt.show()



def main():
    # pred_line = "1.033400, 1.070600, 1.111500, 1.205300, 1.256000, 1.311500, 1.370000, 1.431500, 1.497300, "
    # pred_values = list(map(lambda x: float(x.strip()), filter(lambda x: x.strip(), pred_line.split(','))))
    # print(pred_values)

    file_path = './output/Test_Output.txt'
    predictions, true_values = read_test_output(file_path)

    if len(predictions) == 0:
        print("未解析到有效数据，请检查文件格式！")
        return

    print(f"成功解析 {len(predictions)} 个样本")
    print("首个样本预测值:", predictions[0])
    print("首个样本真实值:", true_values[0])

    plot_mret_distribution(predictions, true_values)
    plot_comparison_at_epsilon(predictions, true_values)


if __name__ == "__main__":
    main()
