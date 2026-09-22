#!/usr/bin/env python3
"""
可视化回环检测结果
绘制全局轨迹，并标注检测到回环的匹配帧并连线
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys
from typing import List, Tuple, Dict


def parse_result_file(filename: str) -> Tuple[Dict[int, np.ndarray], List[Tuple[int, int, float]]]:
    """
    解析回环检测结果文件
    
    Args:
        filename: 结果文件路径
        
    Returns:
        trajectory: 字典，frame_id -> [x, y, z] 或 [x, y, z, qx, qy, qz, qw]
        loop_closures: 列表，[(frame_id1, frame_id2, match_score), ...]
    """
    trajectory = {}
    loop_closures = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    section = None  # 'trajectory' 或 'loop_closures'
    
    for line in lines:
        line = line.strip()
        
        # 跳过空行和注释
        if not line or line.startswith('#'):
            if 'Trajectory' in line:
                section = 'trajectory'
            elif 'Loop Closures' in line or 'Loop Closures' in line:
                section = 'loop_closures'
            continue
        
        if section == 'trajectory':
            # 解析轨迹数据：frame_id x y z [qx qy qz qw]
            parts = line.split()
            if len(parts) >= 4:
                try:
                    frame_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    trajectory[frame_id] = np.array([x, y, z])
                except ValueError:
                    continue
        
        elif section == 'loop_closures':
            # 解析回环数据：frame_id1 frame_id2 [match_score]
            parts = line.split()
            if len(parts) >= 2:
                try:
                    frame_id1 = int(parts[0])
                    frame_id2 = int(parts[1])
                    match_score = float(parts[2]) if len(parts) > 2 else 0.0
                    loop_closures.append((frame_id1, frame_id2, match_score))
                except ValueError:
                    continue
    
    return trajectory, loop_closures


def visualize_3d(trajectory: Dict[int, np.ndarray], 
                 loop_closures: List[Tuple[int, int, float]],
                 output_file: str = None,
                 show_frame_ids: bool = False,
                 show_scores: bool = True,
                 offset_distance: float = 5.0,
                 distance_threshold: float = 10.0,
                 init_elev: float = None,
                 init_azim: float = None,
                 init_dist: float = None):
    """
    3D可视化轨迹和回环
    
    Args:
        trajectory: 轨迹字典
        loop_closures: 回环列表
        output_file: 输出图片文件路径（可选）
        show_frame_ids: 是否显示帧ID
        show_scores: 是否根据匹配得分调整线条粗细和颜色
        offset_distance: 偏移轨迹与原轨迹的距离（米）
        distance_threshold: 判断正确匹配的距离阈值（米），默认10m
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取轨迹点
    frame_ids = sorted(trajectory.keys())
    if not frame_ids:
        print("Error: No trajectory data found!")
        return
    
    traj_points = np.array([trajectory[fid] for fid in frame_ids])
    x = traj_points[:, 0]
    y = traj_points[:, 1]
    z = traj_points[:, 2]
    
    # 计算轨迹的主要方向（用于确定偏移方向）
    # 使用PCA找到主方向
    if len(traj_points) > 1:
        centered = traj_points - traj_points.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        # 主方向是最大特征值对应的特征向量
        main_dir = eigenvecs[:, np.argmax(eigenvals)]
        # 计算垂直于主方向的向量（用于偏移）
        # 选择一个与主方向垂直的方向
        if abs(main_dir[2]) < 0.9:  # 如果主方向不接近z轴
            perp_dir = np.cross(main_dir, np.array([0, 0, 1]))
        else:
            perp_dir = np.cross(main_dir, np.array([1, 0, 0]))
        perp_dir = perp_dir / np.linalg.norm(perp_dir)  # 归一化
    else:
        perp_dir = np.array([1.0, 0.0, 0.0])  # 默认向右偏移
    
    # 计算偏移后的轨迹
    offset_vector = perp_dir * offset_distance
    x_offset = x + offset_vector[0]
    y_offset = y + offset_vector[1]
    z_offset = z + offset_vector[2]
    
    # 如果用户提供了初始视点和视角，就在绘制前设置
    if init_elev is not None or init_azim is not None:
        elev = init_elev if init_elev is not None else ax.elev
        azim = init_azim if init_azim is not None else ax.azim
        ax.view_init(elev=elev, azim=azim)
    if init_dist is not None and hasattr(ax, "dist"):
        ax.dist = init_dist

    # 绘制原始轨迹（更粗）
    ax.plot(x, y, z, 'b-', linewidth=3.0, alpha=0.7, label='Trajectory')
    # 绘制偏移轨迹（更粗）
    ax.plot(x_offset, y_offset, z_offset, 'b--', linewidth=3.0, alpha=0.5, label='Offset Trajectory')
    
    # 创建frame_id到索引的映射
    frame_to_idx = {fid: idx for idx, fid in enumerate(frame_ids)}
    
    # 绘制回环连线（在原轨迹上标出参考关键帧，在平移轨迹上标出匹配关键帧）
    if loop_closures:
        correct_matches = 0
        incorrect_matches = 0
        
        for frame_id1, frame_id2, score in loop_closures:
            if frame_id1 in trajectory and frame_id2 in trajectory:
                # 原始轨迹上的参考关键帧（frame_id1）
                p1 = trajectory[frame_id1]
                
                # 偏移轨迹上对应的匹配关键帧（frame_id2）
                idx2 = frame_to_idx[frame_id2]
                p2_offset = np.array([x_offset[idx2], y_offset[idx2], z_offset[idx2]])
                
                # 计算两帧之间的实际距离
                distance = np.linalg.norm(p1 - trajectory[frame_id2])
                
                # 根据距离判断匹配是否正确
                is_correct = distance <= distance_threshold
                if is_correct:
                    correct_matches += 1
                    line_color = 'green'  # 正确匹配使用绿色
                else:
                    incorrect_matches += 1
                    line_color = 'red'    # 错误匹配使用红色
                
                # 只画一条连接线：从原轨迹的参考关键帧（frame_id1）到平移轨迹的匹配关键帧（frame_id2）
                ax.plot([p1[0], p2_offset[0]], [p1[1], p2_offset[1]], [p1[2], p2_offset[2]],
                       color=line_color, linewidth=2.5, alpha=0.8, linestyle='-', zorder=3)
                
                # 在原轨迹上标注参考关键帧（frame_id1）
                marker_color = 'green' if is_correct else 'red'
                edge_color = 'darkgreen' if is_correct else 'darkred'
                ax.scatter([p1[0]], [p1[1]], [p1[2]],
                          c=marker_color, s=50, marker='o', edgecolors=edge_color, 
                          linewidths=1.5, alpha=0.8, zorder=5, 
                          label='Reference Keyframes' if frame_id1 == loop_closures[0][0] else '')
                
                # 在平移轨迹上标注匹配关键帧（frame_id2）
                ax.scatter([p2_offset[0]], [p2_offset[1]], [p2_offset[2]],
                          c=marker_color, s=50, marker='s', edgecolors=edge_color,  # 使用方形标记区分
                          linewidths=1.5, alpha=0.8, zorder=5,
                          label='Matched Keyframes' if frame_id2 == loop_closures[0][1] else '')
                
                # 可选：显示帧ID
                if show_frame_ids:
                    ax.text(p1[0], p1[1], p1[2], f'{frame_id1}', fontsize=8)
                    ax.text(p2_offset[0], p2_offset[1], p2_offset[2], f'{frame_id2}', fontsize=8)
        
        # 打印统计信息
        print(f"Match Statistics:")
        print(f"  Correct matches (distance <= {distance_threshold}m): {correct_matches}")
        print(f"  Incorrect matches (distance > {distance_threshold}m): {incorrect_matches}")
        print(f"  Total matches: {len(loop_closures)}")
        if len(loop_closures) > 0:
            accuracy = correct_matches / len(loop_closures) * 100
            print(f"  Accuracy: {accuracy:.2f}%")
    
    # 添加回环连线的图例
    if loop_closures:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=3.0, label='Trajectory'),
            Line2D([0], [0], color='blue', linewidth=3.0, linestyle='--', alpha=0.5, label='Offset Trajectory'),
            Line2D([0], [0], color='green', linewidth=2.5, linestyle='-', label=f'Correct Matches'),
            Line2D([0], [0], color='red', linewidth=2.5, linestyle='-', label=f'Incorrect Matches'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=6, markeredgecolor='darkgreen', markeredgewidth=1.5, label='Reference Keyframes (Correct)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                   markersize=6, markeredgecolor='darkgreen', markeredgewidth=1.5, label='Matched Keyframes (Correct)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=6, markeredgecolor='darkred', markeredgewidth=1.5, label='Reference Keyframes (Incorrect)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                   markersize=6, markeredgecolor='darkred', markeredgewidth=1.5, label='Matched Keyframes (Incorrect)')
        ]
        ax.legend(handles=legend_elements, loc='best')
    else:
        ax.legend(loc='best')
    
    # 隐藏坐标轴和网格平面
    ax.set_axis_off()  # 隐藏所有坐标轴、刻度和标签
    ax.grid(False)  # 隐藏网格
    # 隐藏坐标平面背景
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    ax.set_title('Loop Closure Detection Results (3D View)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    # 在终端打印当前视点和视角，便于用户记录/复现
    # 注意：matplotlib 3D Axes 上 elev/azim/dist 都是数值属性
    try:
        curr_elev = getattr(ax, "elev", None)
        curr_azim = getattr(ax, "azim", None)
        curr_dist = getattr(ax, "dist", None)
        print("Current 3D view:")
        print(f"  elev (vertical angle): {curr_elev}")
        print(f"  azim (horizontal angle): {curr_azim}")
        print(f"  dist (camera distance): {curr_dist}")
    except Exception as e:
        print(f"Warning: failed to query current view parameters: {e}")

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved 3D visualization to: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize loop closure detection results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 3D visualization
  python visualize_loop_closure.py result.txt --output result_3d.png
  
  # Show frame IDs
  python visualize_loop_closure.py result.txt --show-frame-ids
        """
    )
    
    parser.add_argument('input_file', type=str, 
                       help='Input result file from loop closure detection')
    parser.add_argument('--output', type=str, default=None,
                       help='Output image file path (optional, if not specified, will display)')
    parser.add_argument('--show-frame-ids', action='store_true',
                       help='Show frame IDs on the plot')
    parser.add_argument('--no-scores', action='store_true',
                       help='Do not use match scores for line styling')
    parser.add_argument('--offset-distance', type=float, default=10.0,
                       help='Distance between original and offset trajectory (default: 10.0 meters)')
    parser.add_argument('--distance-threshold', '-d', type=float, default=20.0,
                       help='Distance threshold for correct match judgment (default: 10.0 meters)')
    parser.add_argument('--elev', type=float, default=None,
                       help='Initial elevation angle of 3D view (degrees, e.g. 30)')
    parser.add_argument('--azim', type=float, default=None,
                       help='Initial azimuth angle of 3D view (degrees, e.g. 60)')
    parser.add_argument('--dist', type=float, default=None,
                       help='Initial camera distance of 3D view (matplotlib Axes3D.dist)')
    
    args = parser.parse_args()
    
    # 解析结果文件
    print(f"Reading result file: {args.input_file}")
    trajectory, loop_closures = parse_result_file(args.input_file)
    
    print(f"Loaded {len(trajectory)} trajectory points")
    print(f"Found {len(loop_closures)} loop closures")
    
    if not trajectory:
        print("Error: No trajectory data found in the file!")
        sys.exit(1)
    
    # 3D可视化
    visualize_3d(trajectory, loop_closures, args.output, 
                 args.show_frame_ids, not args.no_scores, args.offset_distance,
                 args.distance_threshold, args.elev, args.azim, args.dist)
    
    print("Visualization completed!")


if __name__ == '__main__':
    main()

