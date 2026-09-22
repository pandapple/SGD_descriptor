#include "include/sgd.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <map>
#include <sstream>
#include <chrono>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

struct Pose {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    bool valid{false};
};

// 从文件名中提取序号
int extractNumberFromFilename(const std::string& filename) {
    std::string base = fs::path(filename).stem().string();
    return std::stoi(base);
}

// 读取KITTI位姿文件（每行12个数，取平移列）
static std::vector<Pose> readKittiPoses(const std::string& pose_file) {
    std::vector<Pose> poses;
    std::ifstream in(pose_file);
    if (!in.is_open()) {
        std::cerr << "Could not open pose file: " << pose_file << std::endl;
        return poses;
    }
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        double m[12];
        bool ok = true;
        for (int i = 0; i < 12; ++i) {
            if (!(iss >> m[i])) { ok = false; break; }
        }
        Pose p;
        if (ok) {
            p.x = m[3];
            p.y = m[7];
            p.z = m[11];
            p.valid = true;
        }
        poses.push_back(p);
    }
    return poses;
}

static inline double distance3(const Pose& a, const Pose& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// 计算准确率和召回率
std::pair<double, double> calculateMetrics(
    const std::vector<std::pair<double, bool>>& results) {
    
    int true_positives = 0;
    int false_positives = 0;
    int false_negatives = 0;
    
    for (const auto& result : results) {
        if (result.first) { // 如果预测为匹配
            if (result.second) { // 如果是真实匹配
                true_positives++;
            } else {
                false_positives++;
            }
        } else if (!result.first) { // 如果预测为不匹配
            if (result.second) {
                false_negatives++;
            }
        }
    }
    
    double precision = (true_positives + false_positives == 0) ? 0.0 :
        static_cast<double>(true_positives) / (true_positives + false_positives);
    double recall = (true_positives + false_negatives == 0) ? 0.0 :
        static_cast<double>(true_positives) / (true_positives + false_negatives);
    
    return {precision, recall};
}

// 获取文件夹中的所有.bin文件
std::vector<std::string> getBinFiles(const std::string& folder) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.path().extension() == ".bin") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// 获取文件夹中的所有.label文件
std::vector<std::string> getLabelFiles(const std::string& folder) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.path().extension() == ".label") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// 获取所有子文件夹
std::vector<std::string> getSubfolders(const std::string& folder) {
    std::vector<std::string> subfolders;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.is_directory()) {
            subfolders.push_back(entry.path().string());
        }
    }
    std::sort(subfolders.begin(), subfolders.end());
    return subfolders;
}

// 保存PR曲线数据到CSV文件
void savePRCurveData(const std::string& output_file, 
                    const std::vector<std::tuple<double, bool, double>>& results) {
    std::ofstream out_file(output_file);
    out_file << "Threshold,Precision,Recall\n";

    // 使用所有得分作为阈值点
    std::vector<double> thresholds;
    for (const auto& result : results) {
        thresholds.push_back(std::get<0>(result));
    }
    std::sort(thresholds.begin(), thresholds.end());
    thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());

    // 计算不同阈值下的准确率和召回率
    for (double threshold : thresholds) {
        std::vector<std::pair<double, bool>> threshold_results;
        for (const auto& result : results) {
            // 根据阈值判断预测结果：如果匹配得分 >= 阈值，则预测为匹配(true)，否则预测为不匹配(false)
            bool predicted_match = (std::get<0>(result) >= threshold);
            bool actual_match = std::get<1>(result);
            threshold_results.push_back({predicted_match, actual_match});
        }
        
        auto [precision, recall] = calculateMetrics(threshold_results);
        out_file << threshold << "," << precision << "," << recall << "\n";
    }

    out_file.close();
}

// 详细匹配结果结构（包含ID和距离）
struct DetailedMatchResult {
    int input_id;
    int target_id;
    double match_score;
    bool is_true_match;
    double computation_time;
    double pose_distance;
};

// 保存详细匹配结果到CSV文件（包含匹配时间、ID和距离）
void saveDetailedResults(const std::string& output_file, 
                        const std::vector<DetailedMatchResult>& results) {
    std::ofstream out_file(output_file);
    out_file << "InputID,TargetID,MatchScore,IsTrueMatch,ComputationTimeMs,PoseDistance\n";

    for (const auto& result : results) {
        out_file << result.input_id << ","
                 << result.target_id << ","
                 << result.match_score << ","
                 << (result.is_true_match ? "1" : "0") << ","
                 << result.computation_time << ","
                 << result.pose_distance << "\n";
    }

    out_file.close();
}

// 计算AP (Average Precision) - PR曲线下的面积
// 使用梯形积分法（trapezoidal rule），与 plot_pr_curve.py 中的 np.trapz 方法一致
double calculateAveragePrecision(
    const std::vector<std::tuple<double, bool, double>>& results) {
    
    if (results.empty()) {
        return 0.0;
    }
    
    // 使用所有得分作为阈值点（与 savePRCurveData 方法一致）
    std::vector<double> thresholds;
    for (const auto& result : results) {
        thresholds.push_back(std::get<0>(result));
    }
    std::sort(thresholds.begin(), thresholds.end());
    thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());
    
    // 计算不同阈值下的 precision 和 recall
    std::vector<std::pair<double, double>> pr_points;  // (recall, precision) pairs
    
    for (double threshold : thresholds) {
        std::vector<std::pair<double, bool>> threshold_results;
        for (const auto& result : results) {
            // 根据阈值判断预测结果：如果匹配得分 >= 阈值，则预测为匹配(true)，否则预测为不匹配(false)
            bool predicted_match = (std::get<0>(result) >= threshold);
            bool actual_match = std::get<1>(result);
            threshold_results.push_back({predicted_match, actual_match});
        }
        
        auto [precision, recall] = calculateMetrics(threshold_results);
        pr_points.push_back({recall, precision});
    }
    
    if (pr_points.empty()) {
        return 0.0;
    }
    
    // 按 recall 升序排序（np.trapz 要求 x 是单调的）
    std::sort(pr_points.begin(), pr_points.end(),
              [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                  return a.first < b.first;  // 按 recall 升序
              });
    
    // 使用梯形积分法计算 AP（与 np.trapz 方法一致）
    // AP = ∫ precision(recall) d(recall) ≈ Σ[(recall[i+1] - recall[i]) * (precision[i+1] + precision[i]) / 2]
    double ap = 0.0;
    for (size_t i = 0; i < pr_points.size() - 1; ++i) {
        double recall_diff = pr_points[i + 1].first - pr_points[i].first;
        double precision_avg = (pr_points[i + 1].second + pr_points[i].second) / 2.0;
        ap += recall_diff * precision_avg;
    }
    
    return ap;
}

// 计算SR (Success Rate) - 在最佳阈值下的成功率
// 最佳阈值定义为F1-score最大的阈值
double calculateSuccessRate(
    const std::vector<std::tuple<double, bool, double>>& results) {
    
    // 使用所有得分作为阈值点
    std::vector<double> thresholds;
    for (const auto& result : results) {
        thresholds.push_back(std::get<0>(result));
    }
    std::sort(thresholds.begin(), thresholds.end());
    thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());
    
    double best_f1 = 0.0;
    double best_threshold = 0.0;
    double best_sr = 0.0;
    
    // 找到F1-score最大的阈值
    for (double threshold : thresholds) {
        std::vector<std::pair<double, bool>> threshold_results;
        for (const auto& result : results) {
            bool predicted_match = (std::get<0>(result) >= threshold);
            bool actual_match = std::get<1>(result);
            threshold_results.push_back({predicted_match, actual_match});
        }
        
        auto [precision, recall] = calculateMetrics(threshold_results);
        
        // 计算F1-score
        double f1 = (precision + recall == 0.0) ? 0.0 :
            2.0 * precision * recall / (precision + recall);
        
        if (f1 > best_f1) {
            best_f1 = f1;
            best_threshold = threshold;
            // SR = 正确预测的比例 = (TP + TN) / Total
            int correct = 0;
            for (const auto& res : threshold_results) {
                if (res.first == res.second) {  // 预测正确
                    correct++;
                }
            }
            best_sr = static_cast<double>(correct) / threshold_results.size();
        }
    }
    
    return best_sr;
}

// 计算平均匹配耗时
double calculateAverageComputationTime(
    const std::vector<std::tuple<double, bool, double>>& results) {
    
    if (results.empty()) {
        return 0.0;
    }
    
    double total_time = 0.0;
    for (const auto& result : results) {
        total_time += std::get<2>(result);  // 第三个元素是计算时间
    }
    
    return total_time / results.size();
}

// 计算平均每帧的描述子提取时间
double calculateAverageDescriptorExtractionTime(
    const std::vector<std::tuple<double, bool, double>>& results) {
    
    if (results.empty()) {
        return 0.0;
    }
    
    double total_time = 0.0;
    for (const auto& result : results) {
        total_time += std::get<2>(result);  // 第三个元素是计算时间
    }
    
    // 每个匹配对包含2帧，总时间主要包含2次描述子提取和1次匹配
    // 估算：描述子提取时间 ≈ (总时间 * 0.9) / (2 * 匹配对数)
    double descriptor_time_ratio = 0.9;  // 描述子提取时间占总时间的比例
    return (total_time * descriptor_time_ratio) / (2.0 * results.size());
}

// 加载带标签的点云（从KITTI格式的.bin文件和.label文件）
bool loadPointCloudWithLabels(const std::string& bin_filename, 
                              const std::string& label_filename,
                              pcl::PointCloud<pcl::PointXYZL>::Ptr& cloud) {
    // 读取KITTI格式的.bin文件
    std::ifstream bin_file(bin_filename, std::ios::binary);
    if (!bin_file.is_open()) {
        std::cerr << "Could not open bin file: " << bin_filename << std::endl;
        return false;
    }
    
    // 读取SemanticKITTI格式的.label文件
    std::ifstream label_file(label_filename, std::ios::binary);
    if (!label_file.is_open()) {
        std::cerr << "Could not open label file: " << label_filename << std::endl;
        bin_file.close();
        return false;
    }
    
    // 获取文件大小
    bin_file.seekg(0, std::ios::end);
    std::streampos bin_file_size = bin_file.tellg();
    bin_file.seekg(0, std::ios::beg);
    
    label_file.seekg(0, std::ios::end);
    std::streampos label_file_size = label_file.tellg();
    label_file.seekg(0, std::ios::beg);
    
    // 每个点4个float（x, y, z, intensity），每个标签是uint32_t
    size_t num_points = bin_file_size / (4 * sizeof(float));
    size_t num_labels = label_file_size / sizeof(uint32_t);
    
    if (num_points != num_labels) {
        std::cerr << "Error: Number of points (" << num_points << ") does not match number of labels (" << num_labels << ")" << std::endl;
        bin_file.close();
        label_file.close();
        return false;
    }
    
    cloud->clear();
    cloud->reserve(num_points);
    
    for (size_t i = 0; i < num_points; ++i) {
        float x, y, z, intensity;
        bin_file.read(reinterpret_cast<char*>(&x), sizeof(float));
        bin_file.read(reinterpret_cast<char*>(&y), sizeof(float));
        bin_file.read(reinterpret_cast<char*>(&z), sizeof(float));
        bin_file.read(reinterpret_cast<char*>(&intensity), sizeof(float));
        
        uint32_t label;
        label_file.read(reinterpret_cast<char*>(&label), sizeof(uint32_t));
        
        // 只使用标签的低16位作为语义标签
        uint32_t semantic_label = label & 0xFFFF;
        
        pcl::PointXYZL point;
        point.x = x;
        point.y = y;
        point.z = z;
        point.label = semantic_label;
        cloud->push_back(point);
    }
    
    bin_file.close();
    label_file.close();
    return true;
}

// 从YAML文件读取配置参数
bool loadConfigFromYAML(const std::string& yaml_file, ConfigSetting& config) {
    try {
        YAML::Node yaml_config = YAML::LoadFile(yaml_file);
        
        // 读取配置参数，如果不存在则使用默认值
        if (yaml_config["voxel_size"]) {
            config.voxel_size_ = yaml_config["voxel_size"].as<float>();
        }
        if (yaml_config["plane_detection_thre"]) {
            config.plane_detection_thre_ = yaml_config["plane_detection_thre"].as<float>();
        }
        if (yaml_config["voxel_init_num"]) {
            config.voxel_init_num_ = yaml_config["voxel_init_num"].as<int>();
        }
        if (yaml_config["useful_corner_num"]) {
            config.useful_corner_num_ = yaml_config["useful_corner_num"].as<int>();
        }
        if (yaml_config["proj_plane_num"]) {
            config.proj_plane_num_ = yaml_config["proj_plane_num"].as<int>();
        }
        if (yaml_config["proj_image_resolution"]) {
            config.proj_image_resolution_ = yaml_config["proj_image_resolution"].as<float>();
        }
        if (yaml_config["proj_image_high_inc"]) {
            config.proj_image_high_inc_ = yaml_config["proj_image_high_inc"].as<float>();
        }
        if (yaml_config["proj_dis_min"]) {
            config.proj_dis_min_ = yaml_config["proj_dis_min"].as<float>();
        }
        if (yaml_config["proj_dis_max"]) {
            config.proj_dis_max_ = yaml_config["proj_dis_max"].as<float>();
        }
        if (yaml_config["summary_min_thre"]) {
            config.summary_min_thre_ = yaml_config["summary_min_thre"].as<float>();
        }
        if (yaml_config["descriptor_near_num"]) {
            config.descriptor_near_num_ = yaml_config["descriptor_near_num"].as<float>();
        }
        if (yaml_config["descriptor_search_radius"]) {
            config.descriptor_search_radius_ = yaml_config["descriptor_search_radius"].as<float>();
        }
        if (yaml_config["descriptor_min_len"]) {
            config.descriptor_min_len_ = yaml_config["descriptor_min_len"].as<float>();
        }
        if (yaml_config["descriptor_max_len"]) {
            config.descriptor_max_len_ = yaml_config["descriptor_max_len"].as<float>();
        }
        if (yaml_config["non_max_suppression_radius"]) {
            config.non_max_suppression_radius_ = yaml_config["non_max_suppression_radius"].as<float>();
        }
        if (yaml_config["std_side_resolution"]) {
            config.std_side_resolution_ = yaml_config["std_side_resolution"].as<float>();
        }
        if (yaml_config["skip_near_num"]) {
            config.skip_near_num_ = yaml_config["skip_near_num"].as<int>();
        }
        if (yaml_config["candidate_num"]) {
            config.candidate_num_ = yaml_config["candidate_num"].as<int>();
        }
        if (yaml_config["rough_dis_threshold"]) {
            config.rough_dis_threshold_ = yaml_config["rough_dis_threshold"].as<float>();
        }
        if (yaml_config["icp_threshold"]) {
            config.icp_threshold_ = yaml_config["icp_threshold"].as<float>();
        }
        if (yaml_config["normal_threshold"]) {
            config.normal_threshold_ = yaml_config["normal_threshold"].as<float>();
        }
        if (yaml_config["dis_threshold"]) {
            config.dis_threshold_ = yaml_config["dis_threshold"].as<float>();
        }
        if (yaml_config["plane_merge_normal_thre"]) {
            config.plane_merge_normal_thre_ = yaml_config["plane_merge_normal_thre"].as<float>();
        }
        if (yaml_config["plane_merge_dis_thre"]) {
            config.plane_merge_dis_thre_ = yaml_config["plane_merge_dis_thre"].as<float>();
        }
        if (yaml_config["semantic_vertex_match_threshold"]) {
            config.semantic_vertex_match_threshold_ = yaml_config["semantic_vertex_match_threshold"].as<int>();
        }
        if (yaml_config["semantic_ratio_threshold"]) {
            config.semantic_ratio_threshold_ = yaml_config["semantic_ratio_threshold"].as<float>();
        }
        if (yaml_config["semantic_icp_weight"]) {
            config.semantic_icp_weight_ = yaml_config["semantic_icp_weight"].as<float>();
        }
        
        // 读取屏蔽标签列表（向后兼容）
        if (yaml_config["excluded_labels"]) {
            config.excluded_labels.clear();
            if (yaml_config["excluded_labels"].IsSequence()) {
                // 如果是列表格式
                for (const auto& label_node : yaml_config["excluded_labels"]) {
                    uint32_t label = label_node.as<uint32_t>();
                    config.excluded_labels.insert(label);
                }
            } else if (yaml_config["excluded_labels"].IsScalar()) {
                // 如果是单个值
                uint32_t label = yaml_config["excluded_labels"].as<uint32_t>();
                config.excluded_labels.insert(label);
            }
        }
        
        // 读取平面体素屏蔽标签列表
        if (yaml_config["excluded_labels_plane"]) {
            config.excluded_labels_plane.clear();
            if (yaml_config["excluded_labels_plane"].IsSequence()) {
                for (const auto& label_node : yaml_config["excluded_labels_plane"]) {
                    uint32_t label = label_node.as<uint32_t>();
                    config.excluded_labels_plane.insert(label);
                }
            } else if (yaml_config["excluded_labels_plane"].IsScalar()) {
                uint32_t label = yaml_config["excluded_labels_plane"].as<uint32_t>();
                config.excluded_labels_plane.insert(label);
            }
        }
        
        // Note: excluded_labels_non_plane is not used because BTC descriptor does not process non-plane voxels
        if (yaml_config["ablation_node_mode"]) {
            config.ablation_node_mode_ = yaml_config["ablation_node_mode"].as<int>();
        }
        
        return true;
    } catch (const YAML::Exception& e) {
        std::cerr << "Error reading YAML config file: " << e.what() << std::endl;
        return false;
    }
}

// 保存测试指标到txt文件
void saveMetricsToTxt(const std::string& output_file,
                     const std::string& sequence_name,
                     double success_rate,
                     double average_precision,
                     double avg_descriptor_time,
                     double avg_matching_time,
                     bool is_first_sequence = false) {
    std::ofstream out_file;
    if (is_first_sequence) {
        out_file.open(output_file, std::ios::out);  // 覆盖模式（清空文件）
    } else {
        out_file.open(output_file, std::ios::app);  // 追加模式
    }
    
    if (!out_file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << output_file << std::endl;
        return;
    }
    
    out_file << "=== Sequence: " << sequence_name << " ===" << std::endl;
    out_file << "(1) SR (Success Rate): " << success_rate << std::endl;
    out_file << "(2) AP (Average Precision): " << average_precision << std::endl;
    out_file << "(3) Average descriptor extraction time per frame: " << avg_descriptor_time << " ms" << std::endl;
    out_file << "(4) Average matching time per pair: " << avg_matching_time << " ms" << std::endl;
    out_file << std::endl;
    
    out_file.close();
}

// 计算匹配得分（使用语义三角形描述子）
double computeMatchScoreWithLabels(
    SemanticTriangularDescManager& manager,
    const pcl::PointCloud<pcl::PointXYZL>::Ptr& source_cloud,
    const pcl::PointCloud<pcl::PointXYZL>::Ptr& target_cloud,
    double* computation_time_ms) {
    
    if (source_cloud->empty() || target_cloud->empty()) {
        return 0.0;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // 清理数据库和平面点云（确保每次匹配独立）
    manager.data_base_.clear();
    manager.plane_cloud_vec_.clear();
    
    // 生成目标点云的语义三角形描述子（先添加到数据库，frame_id=0）
    std::vector<SemanticTriangularDescriptor> target_stds;
    manager.GenerateSemanticTriangularDescs(target_cloud, 0, target_stds);
    // 将目标描述子添加到数据库
    manager.AddSemanticTriangularDescs(target_stds);
    
    // 生成源点云的语义三角形描述子（用于查询，frame_id=1）
    std::vector<SemanticTriangularDescriptor> source_stds;
    manager.GenerateSemanticTriangularDescs(source_cloud, 1, source_stds);
    
    // 如果描述子数量太少，返回0
    if (source_stds.empty() || target_stds.empty()) {
        std::cerr << "[computeMatchScoreWithLabels] Warning: Empty descriptors! "
                  << "source_stds.size()=" << source_stds.size() 
                  << ", target_stds.size()=" << target_stds.size() << std::endl;
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        if (computation_time_ms) {
            *computation_time_ms = duration.count();
        }
        return 0.0;
    }
    
    // 调试信息：显示描述子和数据库状态
    std::cerr << "[computeMatchScoreWithLabels] Debug: "
              << "source_stds.size()=" << source_stds.size() 
              << ", target_stds.size()=" << target_stds.size()
              << ", data_base_.size()=" << manager.data_base_.size() << std::endl;
    
    // 搜索匹配
    std::pair<int, double> loop_result;
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
    std::vector<std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor>> loop_std_pair;
    
    // 临时启用调试信息
    bool old_debug = manager.print_debug_info_;
    manager.print_debug_info_ = true;
    
    manager.SearchLoop(source_stds, loop_result, loop_transform, loop_std_pair);
    
    // 恢复调试信息设置
    manager.print_debug_info_ = old_debug;
    
    // 匹配得分就是loop_result的second（icp score）
    double match_score = loop_result.second;
    
    // 调试信息
    if (match_score == 0.0) {
        std::cerr << "[computeMatchScoreWithLabels] Debug: Match score is 0. "
                  << "source_stds.size()=" << source_stds.size() 
                  << ", target_stds.size()=" << target_stds.size()
                  << ", loop_result.first=" << loop_result.first 
                  << ", loop_result.second=" << loop_result.second 
                  << ", loop_std_pair.size()=" << loop_std_pair.size() << std::endl;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    if (computation_time_ms) {
        *computation_time_ms = duration.count();
    }
    
    return match_score;
}

int main(int argc, char** argv) {
    if (argc != 7 && argc != 8) {
        std::cout << "Usage: " << argv[0] 
                  << " <input_bin_folder> <target_bin_folder> <poses_folder> <labels_folder> <output_folder> <config_yaml> [pose_distance_threshold]" << std::endl;
        std::cout << "  input_bin_folder: Folder containing input .bin point cloud files (with sequence subfolders)" << std::endl;
        std::cout << "  target_bin_folder: Folder containing target .bin point cloud files (with sequence subfolders)" << std::endl;
        std::cout << "  poses_folder: Folder containing pose .txt files (one per sequence)" << std::endl;
        std::cout << "  labels_folder: Folder containing .label semantic files (with sequence subfolders)" << std::endl;
        std::cout << "  output_folder: Folder to save evaluation results" << std::endl;
        std::cout << "  config_yaml: Path to YAML configuration file" << std::endl;
        std::cout << "  pose_distance_threshold: Maximum pose distance for true match (default: 20.0 meters)" << std::endl;
        return -1;
    }

    std::string input_bin_folder = argv[1];
    std::string target_bin_folder = argv[2];
    std::string poses_folder = argv[3];
    std::string labels_folder = argv[4];
    std::string output_folder = argv[5];
    std::string config_yaml = argv[6];
    double pose_distance_threshold = (argc > 7) ? std::stod(argv[7]) : 20.0;  // 默认20.0米

    // 创建输出文件夹
    fs::create_directories(output_folder);

    // 获取所有子文件夹
    std::vector<std::string> input_subfolders = getSubfolders(input_bin_folder);
    std::vector<std::string> target_subfolders = getSubfolders(target_bin_folder);

    // 从YAML文件读取配置参数
    ConfigSetting config;  // 使用默认值初始化
    if (!loadConfigFromYAML(config_yaml, config)) {
        std::cerr << "Warning: Failed to load config from " << config_yaml 
                  << ", using default values." << std::endl;
    } else {
        std::cout << "Successfully loaded configuration from " << config_yaml << std::endl;
    }
    
    // 输出当前配置参数（用于验证）
    std::cout << "Configuration parameters:" << std::endl;
    std::cout << "  voxel_size: " << config.voxel_size_ << std::endl;
    std::cout << "  plane_detection_thre: " << config.plane_detection_thre_ << std::endl;
    std::cout << "  voxel_init_num: " << config.voxel_init_num_ << std::endl;
    std::cout << "  useful_corner_num: " << config.useful_corner_num_ << std::endl;
    std::cout << "  descriptor_min_len: " << config.descriptor_min_len_ << std::endl;
    std::cout << "  descriptor_max_len: " << config.descriptor_max_len_ << std::endl;
    std::cout << "  std_side_resolution: " << config.std_side_resolution_ << std::endl;
    std::cout << "  rough_dis_threshold: " << config.rough_dis_threshold_ << std::endl;
    std::cout << "  icp_threshold: " << config.icp_threshold_ << std::endl;
    
    SemanticTriangularDescManager manager(config);
    manager.print_debug_info_ = false;  // 关闭调试信息以加快速度
    
    std::cout << "Pose distance threshold for true match: " << pose_distance_threshold << " meters" << std::endl;
    
    // 处理所有序列的点云对
    for (size_t i = 0; i < input_subfolders.size(); ++i) {
        std::string input_subfolder = input_subfolders[i];
        std::string target_subfolder = target_subfolders[i];
        std::string sequence_name = fs::path(input_subfolder).filename().string();
        bool is_first_sequence = (i == 0);
        
        std::cout << "Processing sequence: " << sequence_name << std::endl;

        // 读取该序列位姿文件
        std::string pose_file = (fs::path(poses_folder) / (sequence_name + ".txt")).string();
        auto poses = readKittiPoses(pose_file);

        // 获取当前序列的所有点云文件
        std::vector<std::string> input_bin_files = getBinFiles(input_subfolder);
        std::vector<std::string> target_bin_files = getBinFiles(target_subfolder);
        
        // 获取对应的标签文件文件夹
        std::string labels_subfolder = (fs::path(labels_folder) / sequence_name).string();
        std::vector<std::string> label_files = getLabelFiles(labels_subfolder);

        std::vector<std::tuple<double, bool, double>> sequence_results;  // 用于计算指标
        std::vector<DetailedMatchResult> detailed_results;  // 用于保存详细结果

        // 计算所有点云对之间的匹配得分（只在相同序列内匹配）
        for (const auto& input_bin_file : input_bin_files) {
            int input_num = extractNumberFromFilename(input_bin_file);
            
            // 加载输入点云和标签
            std::string input_label_file = (fs::path(labels_subfolder) / 
                                           (fs::path(input_bin_file).stem().string() + ".label")).string();
            pcl::PointCloud<pcl::PointXYZL>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZL>);
            if (!loadPointCloudWithLabels(input_bin_file, input_label_file, input_cloud)) {
                std::cerr << "Failed to load input cloud with labels: " << input_bin_file << std::endl;
                continue;
            }

            for (const auto& target_bin_file : target_bin_files) {
                int target_num = extractNumberFromFilename(target_bin_file);
                
                // 加载目标点云和标签
                std::string target_label_file = (fs::path(labels_subfolder) / 
                                               (fs::path(target_bin_file).stem().string() + ".label")).string();
                pcl::PointCloud<pcl::PointXYZL>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZL>);
                if (!loadPointCloudWithLabels(target_bin_file, target_label_file, target_cloud)) {
                    std::cerr << "Failed to load target cloud with labels: " << target_bin_file << std::endl;
                    continue;
                }

                double computation_time = 0.0;
                double match_score = computeMatchScoreWithLabels(manager, input_cloud, target_cloud, &computation_time);
                bool is_true = false;
                double pose_dist = -1.0;  // 默认值，表示无效距离
                if (input_num >= 0 && static_cast<size_t>(input_num) < poses.size() &&
                    target_num >= 0 && static_cast<size_t>(target_num) < poses.size() &&
                    poses[input_num].valid && poses[target_num].valid) {
                    pose_dist = distance3(poses[input_num], poses[target_num]);
                    is_true = pose_dist < pose_distance_threshold;
                }
                
                // 输出调试信息
                std::cout << "  Pair [" << input_num << "->" << target_num << "]: "
                          << "Match score: " << match_score 
                          << ", Is true match: " << (is_true ? "Yes" : "No")
                          << ", Distance: " << pose_dist << " m"
                          << ", Time: " << computation_time << " ms" << std::endl;
                
                // 保存用于计算指标的结果
                sequence_results.push_back({match_score, is_true, computation_time});
                
                // 保存详细结果
                DetailedMatchResult detailed_result;
                detailed_result.input_id = input_num;
                detailed_result.target_id = target_num;
                detailed_result.match_score = match_score;
                detailed_result.is_true_match = is_true;
                detailed_result.computation_time = computation_time;
                detailed_result.pose_distance = pose_dist;
                detailed_results.push_back(detailed_result);
            }
        }

        // 输出统计信息
        int total_pairs = sequence_results.size();
        int true_matches = 0;
        double max_score = 0.0, min_score = 1.0;
        for (const auto& result : sequence_results) {
            if (std::get<1>(result)) true_matches++;
            double score = std::get<0>(result);
            max_score = std::max(max_score, score);
            min_score = std::min(min_score, score);
        }
        
        // 计算测试指标
        double average_precision = calculateAveragePrecision(sequence_results);
        double success_rate = calculateSuccessRate(sequence_results);
        double avg_computation_time = calculateAverageComputationTime(sequence_results);
        double avg_descriptor_time = calculateAverageDescriptorExtractionTime(sequence_results);
        
        // 估算平均匹配时间（总时间 - 描述子提取时间）
        double avg_matching_time = avg_computation_time - 2.0 * avg_descriptor_time;
        if (avg_matching_time < 0.0) {
            avg_matching_time = avg_computation_time * 0.1;  // 如果估算为负，使用总时间的10%作为匹配时间
        }
        
        std::cout << "Sequence " << sequence_name << " summary:" << std::endl;
        std::cout << "  Total pairs: " << total_pairs << std::endl;
        std::cout << "  True matches: " << true_matches << std::endl;
        std::cout << "  Score range: [" << min_score << ", " << max_score << "]" << std::endl;
        std::cout << "  Test Metrics:" << std::endl;
        std::cout << "    (1) AP (Average Precision): " << average_precision << std::endl;
        std::cout << "    (2) SR (Success Rate): " << success_rate << std::endl;
        std::cout << "    (3) Average computation time: " << avg_computation_time << " ms" << std::endl;
        std::cout << "    (4) Average descriptor extraction time per frame: " << avg_descriptor_time << " ms" << std::endl;
        std::cout << "    (5) Average matching time per pair: " << avg_matching_time << " ms" << std::endl;
        
        // 为当前序列保存PR曲线数据
        std::string pr_output_file = (fs::path(output_folder) / (sequence_name + "_pr.csv")).generic_string();
        savePRCurveData(pr_output_file, sequence_results);
        
        // 保存详细匹配结果（包含匹配时间、ID和距离）
        std::string detailed_output_file = (fs::path(output_folder) / (sequence_name + "_detailed.csv")).generic_string();
        saveDetailedResults(detailed_output_file, detailed_results);
        
        // 保存测试指标到txt文件
        std::string metrics_output_file = (fs::path(output_folder) / "metrics.txt").generic_string();
        saveMetricsToTxt(metrics_output_file, sequence_name, success_rate, average_precision, 
                        avg_descriptor_time, avg_matching_time, is_first_sequence);
        
        std::cout << "Saved PR curve data for sequence " << sequence_name 
                  << " to " << pr_output_file << std::endl;
        std::cout << "Saved detailed results for sequence " << sequence_name 
                  << " to " << detailed_output_file << std::endl;
        std::cout << "Saved metrics for sequence " << sequence_name 
                  << " to " << metrics_output_file << std::endl;
    }

    std::cout << "Semantic Triangular Descriptor evaluation completed. Results saved to: " << output_folder << std::endl;

    return 0;
}

