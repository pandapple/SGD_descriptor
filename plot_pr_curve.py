import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_pr_curve(csv_file, output_file=None):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制PR曲线
    plt.plot(df['Recall'], df['Precision'], 'b-', linewidth=4)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 设置坐标轴标签
    plt.xlabel('Recall', fontsize=12) #12
    plt.ylabel('Precision', fontsize=12) #12
    
    # 设置标题
    plt.title('Precision-Recall Curve', fontsize=14)
    
    # 设置坐标轴范围
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    
    # 添加一些关键点标记
    #for i in range(0, len(df), len(df)//10):  # 每10%的点标记一次
        #plt.plot(df['Recall'].iloc[i], df['Precision'].iloc[i], 'ro')
        #plt.annotate(f"t={df['Threshold'].iloc[i]:.3f}",
                    #(df['Recall'].iloc[i], df['Precision'].iloc[i]),
                    #xytext=(10, 10), textcoords='offset points')
    
    # 计算并显示AP (Average Precision)
    ap = np.trapz(df['Precision'], df['Recall'])
    plt.text(0.05, 0.05, f'AP = {ap:.3f}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    # 保存图像
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to {output_file}")
    
    # 显示图像
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot PR curve from evaluation results')
    parser.add_argument('input_file', help='Input CSV file containing evaluation results')
    parser.add_argument('--output', '-o', help='Output image file (optional)')
    
    args = parser.parse_args()
    
    plot_pr_curve(args.input_file, args.output) 
