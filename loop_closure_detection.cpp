#include "include/sgd.h"
#include <pcl/filters/voxel_grid.h>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <map>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <set>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fs = std::filesystem;

struct Pose {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    double qx{0.0};
    double qy{0.0};
    double qz{0.0};
    double qw{1.0};
    Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};
    bool has_rotation_matrix{false};
    bool valid{false};
};

// 计算两个位姿之间的欧氏距离
static double computePoseDistance(const Pose& p1, const Pose& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// 检查点云标签是否异常（包含大量unlabeled、outlier、other-object标签）
// 返回true表示标签异常，应该跳过该帧
static bool isLabelAnomalous(const pcl::PointCloud<pcl::PointXYZL>::Ptr& cloud, double threshold = 0.5) {
    if (cloud->empty()) {
        return true;
    }
    
    // 统计异常标签（0: unlabeled, 1: outlier, 99: other-object）的数量
    int anomalous_count = 0;
    int total_count = cloud->size();
    
    for (const auto& point : cloud->points) {
        uint32_t label = point.label & 0xFFFF;  // 只使用低16位作为语义标签
        // 检查是否为异常标签：0 (unlabeled), 1 (outlier), 99 (other-object)
        if (label == 0 || label == 1 || label == 99) {
            anomalous_count++;
        }
    }
    
    // 计算异常标签的比例
    double anomalous_ratio = static_cast<double>(anomalous_count) / total_count;
    
    // 如果异常标签比例超过阈值，认为是异常帧
    return anomalous_ratio > threshold;
}

// 从文件名中提取序号（不含扩展名）
static int extractNumberFromFilename(const std::string& filename) {
    std::string base = fs::path(filename).stem().string();
    try {
        return std::stoi(base);
    } catch (...) {
        return -1;
    }
}

// 获取文件夹中的所有.bin文件，按序号排序
static std::vector<std::string> getBinFiles(const std::string& folder) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.path().extension() == ".bin") {
            files.push_back(entry.path().string());
        }
    }
    // 按文件名中的数字排序
    std::sort(files.begin(), files.end(), [](const std::string& a, const std::string& b) {
        int num_a = extractNumberFromFilename(a);
        int num_b = extractNumberFromFilename(b);
        return num_a < num_b;
    });
    return files;
}

// 获取文件夹中的所有.label文件，按序号排序
static std::vector<std::string> getLabelFiles(const std::string& folder) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.path().extension() == ".label") {
            files.push_back(entry.path().string());
        }
    }
    // 按文件名中的数字排序
    std::sort(files.begin(), files.end(), [](const std::string& a, const std::string& b) {
        int num_a = extractNumberFromFilename(a);
        int num_b = extractNumberFromFilename(b);
        return num_a < num_b;
    });
    return files;
}

// 加载带标签的点云（KITTI格式 .bin + .label）
static bool loadPointCloudWithLabels(const std::string& bin_filename,
                                     const std::string& label_filename,
                                     pcl::PointCloud<pcl::PointXYZL>::Ptr& cloud) {
    std::ifstream bin_file(bin_filename, std::ios::binary);
    if (!bin_file.is_open()) {
        std::cerr << "Could not open bin file: " << bin_filename << std::endl;
        return false;
    }
    std::ifstream label_file(label_filename, std::ios::binary);
    if (!label_file.is_open()) {
        std::cerr << "Could not open label file: " << label_filename << std::endl;
        bin_file.close();
        return false;
    }
    bin_file.seekg(0, std::ios::end);
    std::streampos bin_file_size = bin_file.tellg();
    bin_file.seekg(0, std::ios::beg);
    label_file.seekg(0, std::ios::end);
    std::streampos label_file_size = label_file.tellg();
    label_file.seekg(0, std::ios::beg);
    size_t num_points = bin_file_size / (4 * sizeof(float));
    size_t num_labels = label_file_size / sizeof(uint32_t);
    if (num_points != num_labels) {
        std::cerr << "Error: point count (" << num_points << ") != label count (" << num_labels << ")" << std::endl;
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
        uint32_t semantic_label = label & 0xFFFF;
        pcl::PointXYZL point;
        point.x = x; point.y = y; point.z = z;
        point.label = semantic_label;
        cloud->push_back(point);
    }
    bin_file.close();
    label_file.close();
    return true;
}

// 将Pose转换为4x4变换矩阵（用于位姿误差计算）
static Eigen::Matrix4d poseToMatrix(const Pose& p) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(0, 3) = p.x; T(1, 3) = p.y; T(2, 3) = p.z;
    if (p.has_rotation_matrix) {
        T.block<3, 3>(0, 0) = p.rotation;
    } else if (p.qw != 1.0 || p.qx != 0.0 || p.qy != 0.0 || p.qz != 0.0) {
        Eigen::Quaterniond q(p.qw, p.qx, p.qy, p.qz);
        T.block<3, 3>(0, 0) = q.toRotationMatrix();
    }
    return T;
}

// 相对位姿 T_rel = T2^(-1) * T1（从 frame1 到 frame2）
static Eigen::Matrix4d computeRelativePose(const Eigen::Matrix4d& T1, const Eigen::Matrix4d& T2) {
    return T2.inverse() * T1;
}

// 将估计位姿从点云坐标系变换到 KITTI 位姿坐标系（与 pose_est.cpp 一致）
static void transform_pose_for_kitti(Eigen::Matrix4d& T) {
    Eigen::Matrix3d T_swap;
    T_swap << 0, -1, 0, 0, 0, -1, 1, 0, 0;
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.block<3, 1>(0, 3);
    T.block<3, 3>(0, 0) = T_swap * R * T_swap.transpose();
    T.block<3, 1>(0, 3) = Eigen::Vector3d(-t.y(), -t.z(), t.x());
}

// 计算相对位姿的平移与旋转误差
static void computePoseError(const Eigen::Matrix4d& T_est, const Eigen::Matrix4d& T_gt,
                             double& trans_error, double& rot_error) {
    Eigen::Vector3d t_est = T_est.block<3, 1>(0, 3);
    Eigen::Vector3d t_gt = T_gt.block<3, 1>(0, 3);
    trans_error = (t_est - t_gt).norm();
    Eigen::Matrix3d R_est = T_est.block<3, 3>(0, 0);
    Eigen::Matrix3d R_gt = T_gt.block<3, 3>(0, 0);
    Eigen::AngleAxisd aa(R_est * R_gt.transpose());
    rot_error = std::abs(aa.angle()) * 180.0 / M_PI;
}

// 读取位姿文件（支持KITTI格式：每行12个数，或timestamp x y z qx qy qz qw格式）
static std::vector<Pose> readPoseFile(const std::string& pose_file) {
    std::vector<Pose> poses;
    std::ifstream in(pose_file);
    if (!in.is_open()) {
        std::cerr << "Could not open pose file: " << pose_file << std::endl;
        return poses;
    }

    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::vector<double> values;
        double val;
        while (iss >> val) {
            values.push_back(val);
        }
        
        Pose p;
        if (values.size() == 12) {
            // KITTI格式：3x4矩阵，同时保存旋转和平移
            p.x = values[3];
            p.y = values[7];
            p.z = values[11];
            p.rotation << values[0], values[1], values[2],
                          values[4], values[5], values[6],
                          values[8], values[9], values[10];
            p.has_rotation_matrix = true;
            p.valid = true;
        } else if (values.size() == 8) {
            // timestamp x y z qx qy qz qw格式
            p.x = values[1];
            p.y = values[2];
            p.z = values[3];
            p.qx = values[4];
            p.qy = values[5];
            p.qz = values[6];
            p.qw = values[7];
            p.valid = true;
        } else if (values.size() >= 3) {
            // 简单格式：x y z
            p.x = values[0];
            p.y = values[1];
            p.z = values[2];
            p.valid = true;
        }
        poses.push_back(p);
    }
    return poses;
}

// 保存轨迹和回环检测结果
static void saveResults(const std::string& output_file,
                       const std::vector<Pose>& poses,
                       const std::vector<std::pair<int, int>>& loop_pairs,
                       const std::vector<double>& match_scores,
                       const std::vector<int>& is_correct_flags) {
    std::ofstream out(output_file);
    out << "# Loop Closure Detection Results (Semantic Triangular Descriptor)" << std::endl;
    out << "# Format: frame_id x y z [qx qy qz qw] loop_frame_id match_score pose_distance is_correct(1=correct,0=wrong)" << std::endl;
    out << "# Trajectory:" << std::endl;
    
    for (size_t i = 0; i < poses.size(); ++i) {
        if (poses[i].valid) {
            out << i << " " << poses[i].x << " " << poses[i].y << " " << poses[i].z;
            if (poses[i].qw != 1.0 || poses[i].qx != 0.0 || poses[i].qy != 0.0 || poses[i].qz != 0.0) {
                out << " " << poses[i].qx << " " << poses[i].qy << " " << poses[i].qz << " " << poses[i].qw;
            }
            out << std::endl;
        }
    }
    
    out << "# Loop Closures:" << std::endl;
    for (size_t i = 0; i < loop_pairs.size(); ++i) {
        int frame_id1 = loop_pairs[i].first;
        int frame_id2 = loop_pairs[i].second;
        int is_correct = (i < is_correct_flags.size()) ? is_correct_flags[i] : 0;
        out << frame_id1 << " " << frame_id2;
        if (i < match_scores.size()) {
            out << " " << match_scores[i];
        }
        
        // 计算并写入两帧的距离
        double pose_distance = -1.0;  // 默认值表示无效
        if (frame_id1 >= 0 && frame_id1 < (int)poses.size() && poses[frame_id1].valid &&
            frame_id2 >= 0 && frame_id2 < (int)poses.size() && poses[frame_id2].valid) {
            pose_distance = computePoseDistance(poses[frame_id1], poses[frame_id2]);
        }
        out << " " << pose_distance << " " << is_correct;
        
        out << std::endl;
    }
    
    out.close();
}

static double elapsedMs(const std::chrono::high_resolution_clock::time_point& t0,
                      const std::chrono::high_resolution_clock::time_point& t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct PairTimingRecord {
    int query_frame{0};
    int candidate_frame{0};
    double total_ms{0.0};
};

static void saveFrameTimingCsv(const std::string& path,
                               const std::vector<std::pair<int, double>>& records) {
    std::ofstream out(path);
    out << "frame_idx,total_ms\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& r : records) {
        out << r.first << "," << r.second << "\n";
    }
    out.close();
}

static void savePairTimingCsv(const std::string& path, const std::vector<PairTimingRecord>& records) {
    std::ofstream out(path);
    out << "query_frame,candidate_frame,total_ms\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& r : records) {
        out << r.query_frame << "," << r.candidate_frame << "," << r.total_ms << "\n";
    }
    out.close();
}

// 从YAML文件读取配置参数（语义三角形描述子 / BTC 兼容）
bool loadConfigFromYAML(const std::string& yaml_file, ConfigSetting& config) {
    try {
        YAML::Node yaml_config = YAML::LoadFile(yaml_file);
        if (yaml_config["cloud_ds_size"]) config.cloud_ds_size_ = yaml_config["cloud_ds_size"].as<double>();
        if (yaml_config["voxel_size"]) config.voxel_size_ = yaml_config["voxel_size"].as<float>();
        if (yaml_config["plane_detection_thre"]) config.plane_detection_thre_ = yaml_config["plane_detection_thre"].as<float>();
        if (yaml_config["voxel_init_num"]) config.voxel_init_num_ = yaml_config["voxel_init_num"].as<int>();
        if (yaml_config["useful_corner_num"]) config.useful_corner_num_ = yaml_config["useful_corner_num"].as<int>();
        if (yaml_config["proj_plane_num"]) config.proj_plane_num_ = yaml_config["proj_plane_num"].as<int>();
        if (yaml_config["proj_image_resolution"]) config.proj_image_resolution_ = yaml_config["proj_image_resolution"].as<float>();
        if (yaml_config["proj_image_high_inc"]) config.proj_image_high_inc_ = yaml_config["proj_image_high_inc"].as<float>();
        if (yaml_config["proj_dis_min"]) config.proj_dis_min_ = yaml_config["proj_dis_min"].as<float>();
        if (yaml_config["proj_dis_max"]) config.proj_dis_max_ = yaml_config["proj_dis_max"].as<float>();
        if (yaml_config["summary_min_thre"]) config.summary_min_thre_ = yaml_config["summary_min_thre"].as<float>();
        if (yaml_config["line_filter_enable"]) config.line_filter_enable_ = yaml_config["line_filter_enable"].as<int>();
        if (yaml_config["descriptor_near_num"]) config.descriptor_near_num_ = yaml_config["descriptor_near_num"].as<float>();
        if (yaml_config["descriptor_search_radius"]) config.descriptor_search_radius_ = yaml_config["descriptor_search_radius"].as<float>();
        if (yaml_config["descriptor_min_len"]) config.descriptor_min_len_ = yaml_config["descriptor_min_len"].as<float>();
        if (yaml_config["descriptor_max_len"]) config.descriptor_max_len_ = yaml_config["descriptor_max_len"].as<float>();
        if (yaml_config["non_max_suppression_radius"]) config.non_max_suppression_radius_ = yaml_config["non_max_suppression_radius"].as<float>();
        if (yaml_config["std_side_resolution"]) config.std_side_resolution_ = yaml_config["std_side_resolution"].as<float>();
        if (yaml_config["skip_near_num"]) config.skip_near_num_ = yaml_config["skip_near_num"].as<int>();
        if (yaml_config["candidate_num"]) config.candidate_num_ = yaml_config["candidate_num"].as<int>();
        if (yaml_config["rough_dis_threshold"]) config.rough_dis_threshold_ = yaml_config["rough_dis_threshold"].as<float>();
        if (yaml_config["icp_threshold"]) config.icp_threshold_ = yaml_config["icp_threshold"].as<float>();
        if (yaml_config["normal_threshold"]) config.normal_threshold_ = yaml_config["normal_threshold"].as<float>();
        if (yaml_config["dis_threshold"]) config.dis_threshold_ = yaml_config["dis_threshold"].as<float>();
        if (yaml_config["plane_merge_normal_thre"]) config.plane_merge_normal_thre_ = yaml_config["plane_merge_normal_thre"].as<float>();
        if (yaml_config["plane_merge_dis_thre"]) config.plane_merge_dis_thre_ = yaml_config["plane_merge_dis_thre"].as<float>();
        if (yaml_config["semantic_vertex_match_threshold"]) config.semantic_vertex_match_threshold_ = yaml_config["semantic_vertex_match_threshold"].as<int>();
        if (yaml_config["semantic_ratio_threshold"]) config.semantic_ratio_threshold_ = yaml_config["semantic_ratio_threshold"].as<float>();
        if (yaml_config["semantic_icp_weight"]) config.semantic_icp_weight_ = yaml_config["semantic_icp_weight"].as<float>();
        if (yaml_config["excluded_labels"]) {
            config.excluded_labels.clear();
            if (yaml_config["excluded_labels"].IsSequence()) {
                for (const auto& n : yaml_config["excluded_labels"]) config.excluded_labels.insert(n.as<uint32_t>());
            } else {
                config.excluded_labels.insert(yaml_config["excluded_labels"].as<uint32_t>());
            }
        }
        if (yaml_config["excluded_labels_plane"]) {
            config.excluded_labels_plane.clear();
            if (yaml_config["excluded_labels_plane"].IsSequence()) {
                for (const auto& n : yaml_config["excluded_labels_plane"]) config.excluded_labels_plane.insert(n.as<uint32_t>());
            } else {
                config.excluded_labels_plane.insert(yaml_config["excluded_labels_plane"].as<uint32_t>());
            }
        }
        if (yaml_config["ablation_node_mode"]) config.ablation_node_mode_ = yaml_config["ablation_node_mode"].as<int>();
        return true;
    } catch (const YAML::Exception& e) {
        std::cerr << "Error reading YAML config: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char** argv) {
    // 用法：loop_closure_detection <bin_folder> <label_folder> <pose_file> <output_folder> <config_yaml> ...
    if (argc < 6) {
        std::cout << "Usage: " << argv[0]
                  << " <bin_folder> <label_folder> <pose_file> <output_folder> <config_yaml> [skip_near_frames] [match_threshold] [max_candidate_frames] [pose_distance_threshold] [keyframe_interval] [is_kitti]" << std::endl;
        std::cout << "  bin_folder: folder containing .bin point cloud files" << std::endl;
        std::cout << "  label_folder: folder containing .label semantic label files" << std::endl;
        std::cout << "  pose_file: text file containing poses (KITTI format or timestamp x y z qx qy qz qw)" << std::endl;
        std::cout << "  output_folder: directory for loop results and timing txt files" << std::endl;
        std::cout << "  config_yaml: path to YAML configuration file" << std::endl;
        std::cout << "  skip_near_frames: minimum frame difference for loop detection (default: 150)" << std::endl;
        std::cout << "  match_threshold: minimum match score for loop closure (default: 0.1)" << std::endl;
        std::cout << "  max_candidate_frames: maximum number of frames in candidate queue (default: 1000, 0=unlimited)" << std::endl;
        std::cout << "  pose_distance_threshold: maximum pose distance for candidate selection (default: 30.0 meters)" << std::endl;
        std::cout << "  keyframe_interval: interval between keyframes (default: 10)" << std::endl;
        std::cout << "  is_kitti: 1 for KITTI dataset (apply coordinate transform to estimated pose), 0 for others (default: 0)" << std::endl;
        return -1;
    }

    std::string bin_folder = argv[1];
    std::string label_folder = argv[2];
    std::string pose_file = argv[3];
    std::string output_folder = argv[4];
    std::string config_yaml = argv[5];
    int skip_near_frames = (argc > 6) ? std::stoi(argv[6]) : 150;
    double match_threshold = (argc > 7) ? std::stod(argv[7]) : 0.3;
    size_t max_candidate_frames = (argc > 8) ? std::stoul(argv[8]) : 1000;
    double pose_distance_threshold = (argc > 9) ? std::stod(argv[9]) : 50.0;
    int keyframe_interval = (argc > 10) ? std::stoi(argv[10]) : 10;
    bool is_kitti = (argc > 11) ? (std::stoi(argv[11]) != 0) : true;

    std::cout << "Loop Closure Detection (Semantic Triangular Descriptor, Pose-based Filtering)" << std::endl;
    std::cout << "  Bin folder: " << bin_folder << std::endl;
    std::cout << "  Label folder: " << label_folder << std::endl;
    std::cout << "  Pose file: " << pose_file << std::endl;
    std::cout << "  Output folder: " << output_folder << std::endl;
    std::cout << "  Config file: " << config_yaml << std::endl;
    std::cout << "  Skip near frames: " << skip_near_frames << std::endl;
    std::cout << "  Match threshold: " << match_threshold << std::endl;
    std::cout << "  Max candidate frames: " << (max_candidate_frames == 0 ? "unlimited" : std::to_string(max_candidate_frames)) << std::endl;
    std::cout << "  Pose distance threshold: " << pose_distance_threshold << " meters" << std::endl;
    std::cout << "  Keyframe interval: " << keyframe_interval << std::endl;
    std::cout << "  is_kitti (coordinate transform): " << (is_kitti ? "yes" : "no") << std::endl;

    fs::create_directories(output_folder);
    const std::string results_file = (fs::path(output_folder) / "loop_closure_results.txt").string();
    const std::string detail_file = (fs::path(output_folder) / "matches.detail.txt").string();
    const std::string desc_timing_file = (fs::path(output_folder) / "desc_timing.txt").string();
    const std::string match_timing_file = (fs::path(output_folder) / "match_timing.txt").string();
    const std::string pose_timing_file = (fs::path(output_folder) / "pose_timing.txt").string();

    // 读取位姿
    std::vector<Pose> poses = readPoseFile(pose_file);
    std::cout << "Loaded " << poses.size() << " poses" << std::endl;

    // 获取所有.bin文件和.label文件
    std::vector<std::string> bin_files = getBinFiles(bin_folder);
    std::vector<std::string> label_files = getLabelFiles(label_folder);
    std::cout << "Found " << bin_files.size() << " point cloud files" << std::endl;
    std::cout << "Found " << label_files.size() << " label files" << std::endl;

    if (bin_files.empty()) {
        std::cerr << "No .bin files found in " << bin_folder << std::endl;
        return -1;
    }
    if (label_files.empty()) {
        std::cerr << "No .label files found in " << label_folder << std::endl;
        return -1;
    }
    if (bin_files.size() != label_files.size()) {
        std::cerr << "Warning: Number of .bin files (" << bin_files.size() 
                  << ") does not match number of .label files (" << label_files.size() << ")" << std::endl;
    }

    // 初始化语义三角形描述子管理器
    ConfigSetting config;
    if (!loadConfigFromYAML(config_yaml, config)) {
        std::cerr << "Warning: Failed to load config from " << config_yaml 
                  << ", using default values." << std::endl;
    } else {
        std::cout << "Successfully loaded configuration from " << config_yaml << std::endl;
    }
    SemanticTriangularDescManager manager(config);

    // 存储回环检测结果
    std::vector<std::pair<int, int>> loop_pairs;
    std::vector<double> match_scores;
    std::vector<int> loop_is_correct_flags;
    std::map<int, std::vector<std::pair<int, double>>> frame_matches;  // frame_id -> [(matched_frame_id, score), ...]
    struct MatchResult {
        int candidate_frame_id;
        double match_score;
        double pose_distance;
    };
    std::map<int, std::vector<MatchResult>> all_filtered_matches;  // frame_id -> [MatchResult, ...]
    std::vector<double> translation_errors;
    std::vector<double> rotation_errors;
    
    // 统计回环匹配结果
    int total_loop_matches = 0;  // 总回环匹配对数
    int correct_loop_matches = 0;  // 正确匹配对数（位姿估计平移误差 < 2*voxel_size）
    const double correct_trans_error_threshold = 2.0 * static_cast<double>(config.voxel_size_);  // 正确：估计平移误差 < 2*voxel_size

    std::vector<std::pair<int, double>> desc_timing_records;  // 有成功匹配的查询帧描述子耗时
    std::vector<PairTimingRecord> match_timing_records;       // 成功匹配对的检索匹配耗时
    std::vector<PairTimingRecord> pose_timing_records;        // 成功匹配对的几何验证/位姿耗时

    // 使用统一的 manager，所有历史帧的描述子都存储在 data_base_ 中
    // 维护 frame_id 映射：外部 frame_id -> manager 内部 frame_id（即 plane_cloud_vec_ 索引）
    std::map<int, int> frame_id_to_manager_frame_id;
    std::vector<int> candidate_frame_ids;  // 保持顺序，用于FIFO管理队列大小
    std::set<int> anomalous_frame_ids;  // 记录标签异常的帧ID，这些帧不参与回环检测

    std::cout << "\nProcessing frames (incremental SLAM mode with keyframes)..." << std::endl;
    std::cout << "Keyframe interval: " << keyframe_interval << " frames" << std::endl;
    auto total_start = std::chrono::high_resolution_clock::now();
    int keyframe_count = 0;

    // 增量式处理：逐帧处理，只有关键帧才进行回环检测
    for (size_t i = 0; i < bin_files.size(); ++i) {
        int current_frame_id = extractNumberFromFilename(bin_files[i]);
        if (current_frame_id < 0) {
            current_frame_id = i;
        }

        // 判断是否为关键帧（每10帧取一帧）
        bool is_keyframe = (current_frame_id % keyframe_interval == 0);
        
        // 非关键帧跳过处理（不进行描述子生成和匹配）
        if (!is_keyframe) {
            continue;
        }
        
        keyframe_count++;

        // 只在需要时输出进度信息（减少I/O开销）
        bool show_progress = (keyframe_count <= 5 || keyframe_count % 10 == 0 || i + 1 == bin_files.size());
        
        // 步骤1：先使用pose真值筛选候选帧（在读取点云之前，节省运算时间）
        std::vector<std::pair<int, double>> pose_filtered_candidates;  // (frame_id, distance)
        bool has_valid_pose = (current_frame_id < (int)poses.size() && poses[current_frame_id].valid);
        
        if (has_valid_pose && !candidate_frame_ids.empty()) {
            const Pose& current_pose = poses[current_frame_id];
            
            // 使用pose真值筛选候选帧（距离小于pose_distance_threshold的帧）
            for (int candidate_frame_id : candidate_frame_ids) {
                // 跳过距离当前帧太近的帧
                if (std::abs(candidate_frame_id - current_frame_id) < skip_near_frames) {
                    continue;
                }
                
                // 检查pose是否有效
                if (candidate_frame_id >= (int)poses.size() || !poses[candidate_frame_id].valid) {
                    continue;
                }
                
                // 计算pose距离
                double pose_distance = computePoseDistance(current_pose, poses[candidate_frame_id]);
                
                // 只保留距离小于阈值的帧
                if (pose_distance < pose_distance_threshold) {
                    pose_filtered_candidates.push_back({candidate_frame_id, pose_distance});
                }
            }
        }
        
        // 步骤2：读取当前帧点云和生成描述子
        // 注意：即使没有候选帧，也需要读取点云和生成描述子，以便将当前帧添加到数据库（用于后续帧的匹配）
        // 但通过先进行pose筛选，我们可以跳过不必要的匹配计算，节省时间
        // 查找对应的标签文件
        std::string label_file;
        for (const auto& lf : label_files) {
            int label_frame_id = extractNumberFromFilename(lf);
            if (label_frame_id == current_frame_id) {
                label_file = lf;
                break;
            }
        }
        
        if (label_file.empty()) {
            std::cerr << "Warning: No label file found for frame " << current_frame_id << std::endl;
            continue;
        }

        // 读取当前帧点云（带语义标签）
        pcl::PointCloud<pcl::PointXYZL>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZL>);
        if (!loadPointCloudWithLabels(bin_files[i], label_file, current_cloud)) {
            std::cerr << "Failed to load: " << bin_files[i] << " with " << label_file << std::endl;
            continue;
        }

        // 检查当前帧的标签是否异常（包含大量unlabeled、outlier、other-object）
        if (isLabelAnomalous(current_cloud)) {
            if (show_progress) {
                std::cout << "Skipping keyframe " << current_frame_id 
                          << " due to anomalous labels (too many unlabeled/outlier/other-object)" << std::endl;
            }
            anomalous_frame_ids.insert(current_frame_id);
            continue;
        }

        // 生成当前帧的语义三角形描述子（GenerateSemanticTriangularDescs 会 push plane_cloud 和 semantic_voxel_map）
        int current_manager_frame_id = static_cast<int>(manager.plane_cloud_vec_.size());
        std::vector<SemanticTriangularDescriptor> current_stds;
        auto desc_t0 = std::chrono::high_resolution_clock::now();
        manager.GenerateSemanticTriangularDescs(current_cloud, current_manager_frame_id, current_stds);
        auto desc_t1 = std::chrono::high_resolution_clock::now();
        const double desc_ms = elapsedMs(desc_t0, desc_t1);
        bool frame_has_successful_match = false;
        
        if (show_progress) {
            std::cout << "Processing keyframe " << current_frame_id << " (keyframe #" << keyframe_count 
                      << ", frame " << (i+1) << "/" << bin_files.size() 
                      << "), candidate queue: " << candidate_frame_ids.size() 
                      << ", descriptors: " << current_stds.size();
            if (!pose_filtered_candidates.empty()) {
                std::cout << ", pose-filtered candidates: " << pose_filtered_candidates.size();
            }
            std::cout << "...";
        }

        // 步骤3：对筛选后的候选帧进行匹配，选择得分最高的
        double best_match_score = 0.0;
        int best_match_frame = -1;
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> best_loop_transform;
        best_loop_transform.first = Eigen::Vector3d::Zero();
        best_loop_transform.second = Eigen::Matrix3d::Identity();
        
        if (!current_stds.empty() && !pose_filtered_candidates.empty()) {
            for (const auto& candidate : pose_filtered_candidates) {
                int candidate_frame_id = candidate.first;
                
                if (anomalous_frame_ids.find(candidate_frame_id) != anomalous_frame_ids.end()) {
                    continue;
                }
                
                auto it = frame_id_to_manager_frame_id.find(candidate_frame_id);
                if (it == frame_id_to_manager_frame_id.end()) {
                    continue;
                }
                int candidate_manager_frame_id = it->second;
                
                // 从全局数据库中提取该候选帧的所有描述子（按 frame_number_ 筛选）
                std::vector<SemanticTriangularDescriptor> candidate_stds;
                for (const auto& kv : manager.data_base_) {
                    for (const auto& desc : kv.second) {
                        if (desc.frame_number_ == static_cast<unsigned short>(candidate_manager_frame_id)) {
                            candidate_stds.push_back(desc);
                        }
                    }
                }
                
                if (candidate_stds.empty()) {
                    continue;
                }
                
                // 创建临时 manager：仅包含该候选帧的 plane_cloud、semantic_voxel_map 和描述子
                SemanticTriangularDescManager temp_manager(config);
                if (candidate_manager_frame_id < static_cast<int>(manager.plane_cloud_vec_.size())) {
                    temp_manager.plane_cloud_vec_.push_back(manager.plane_cloud_vec_[candidate_manager_frame_id]);
                } else {
                    continue;
                }
                if (!manager.plane_cloud_vec_.empty()) {
                    temp_manager.plane_cloud_vec_.push_back(manager.plane_cloud_vec_.back());
                }
                if (candidate_manager_frame_id < static_cast<int>(manager.semantic_voxel_map_vec_.size())) {
                    temp_manager.semantic_voxel_map_vec_.push_back(manager.semantic_voxel_map_vec_[candidate_manager_frame_id]);
                }
                if (!manager.semantic_voxel_map_vec_.empty()) {
                    temp_manager.semantic_voxel_map_vec_.push_back(manager.semantic_voxel_map_vec_.back());
                }
                
                for (auto& d : candidate_stds) {
                    d.frame_number_ = 0;
                }
                temp_manager.AddSemanticTriangularDescs(candidate_stds);
                
                std::pair<int, double> search_result(-1, 0);
                std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
                std::vector<std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor>> loop_std_pair;
                double selector_ms = 0.0;
                double verify_ms = 0.0;
                temp_manager.SearchLoop(current_stds, search_result, loop_transform, loop_std_pair,
                                        &selector_ms, &verify_ms);
                double score = search_result.second;
                
                double pose_dist = candidate.second;
                MatchResult match_result;
                match_result.candidate_frame_id = candidate_frame_id;
                match_result.match_score = score;
                match_result.pose_distance = pose_dist;
                all_filtered_matches[current_frame_id].push_back(match_result);
                
                if (score > match_threshold) {
                    frame_has_successful_match = true;
                    match_timing_records.push_back({current_frame_id, candidate_frame_id, selector_ms});
                    pose_timing_records.push_back({current_frame_id, candidate_frame_id, verify_ms});
                    frame_matches[current_frame_id].push_back({candidate_frame_id, score});
                    if (score > best_match_score) {
                        best_match_score = score;
                        best_match_frame = candidate_frame_id;
                        best_loop_transform = loop_transform;
                    }
                }
            }
        }

        if (frame_has_successful_match) {
            desc_timing_records.push_back({current_frame_id, desc_ms});
        }
        
        // 将当前帧描述子加入全局 manager（用于后续帧的匹配）
        if (!current_stds.empty()) {
            frame_id_to_manager_frame_id[current_frame_id] = current_manager_frame_id;
            manager.AddSemanticTriangularDescs(current_stds);
            candidate_frame_ids.push_back(current_frame_id);
            
            if (max_candidate_frames > 0 && candidate_frame_ids.size() > max_candidate_frames) {
                int oldest_frame = candidate_frame_ids.front();
                candidate_frame_ids.erase(candidate_frame_ids.begin());
                frame_id_to_manager_frame_id.erase(oldest_frame);
            }
        }

        if (show_progress) {
            if (best_match_frame >= 0) {
                std::cout << " Loop detected with frame " << best_match_frame 
                          << " (score: " << best_match_score << ")" << std::endl;
            } else {
                std::cout << " No loop detected" << std::endl;
            }
        }
        
        if (best_match_frame >= 0) {
            loop_pairs.push_back({current_frame_id, best_match_frame});
            match_scores.push_back(best_match_score);
            total_loop_matches++;
            int is_correct = 0;
            
            if (has_valid_pose && best_match_frame < (int)poses.size() && poses[best_match_frame].valid &&
                poses[current_frame_id].has_rotation_matrix && poses[best_match_frame].has_rotation_matrix) {
                Eigen::Matrix4d T_cur = poseToMatrix(poses[current_frame_id]);
                Eigen::Matrix4d T_matched = poseToMatrix(poses[best_match_frame]);
                Eigen::Matrix4d T_gt_rel = computeRelativePose(T_cur, T_matched);
                Eigen::Matrix4d T_est_rel = Eigen::Matrix4d::Identity();
                T_est_rel.block<3, 3>(0, 0) = best_loop_transform.second;
                T_est_rel.block<3, 1>(0, 3) = best_loop_transform.first;
                if (is_kitti) {
                    transform_pose_for_kitti(T_est_rel);
                }
                double te, re;
                computePoseError(T_est_rel, T_gt_rel, te, re);
                if (te < correct_trans_error_threshold) {
                    correct_loop_matches++;
                    translation_errors.push_back(te);
                    rotation_errors.push_back(re);
                    is_correct = 1;
                }
            }
            loop_is_correct_flags.push_back(is_correct);
        }

        // 每10个关键帧输出一次详细进度
        if (keyframe_count % 10 == 0 && keyframe_count > 0) {
            std::cout << "Progress: " << keyframe_count << " keyframes processed, " 
                      << loop_pairs.size() << " loops detected" << std::endl;
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    
    std::cout << "\nLoop closure detection completed!" << std::endl;
    std::cout << "  Total frames: " << bin_files.size() << std::endl;
    std::cout << "  Keyframes processed: " << keyframe_count << std::endl;
    std::cout << "  Total loops detected: " << loop_pairs.size() << std::endl;
    std::cout << "  Total processing time: " << total_duration.count() / 1000.0 << " seconds" << std::endl;
    if (keyframe_count > 0) {
        std::cout << "  Average time per keyframe: " << (total_duration.count() / (double)keyframe_count) << " ms" << std::endl;
    }
    
    // 输出回环匹配统计信息
    std::cout << "\nLoop Closure Match Statistics:" << std::endl;
    std::cout << "  Total loop matches: " << total_loop_matches << std::endl;
    std::cout << "  Correct loop matches (trans error < 2*voxel_size=" << correct_trans_error_threshold << " m): " << correct_loop_matches << std::endl;
    if (total_loop_matches > 0) {
        double correct_rate = (double)correct_loop_matches / total_loop_matches * 100.0;
        std::cout << "  Match accuracy: " << std::fixed << std::setprecision(2) << correct_rate << "%" << std::endl;
    } else {
        std::cout << "  Match accuracy: N/A (no loop matches detected)" << std::endl;
    }

    if (!translation_errors.empty()) {
        double trans_rmse = 0.0, rot_rmse = 0.0, trans_mean = 0.0, rot_mean = 0.0, trans_max = 0.0, rot_max = 0.0;
        for (size_t i = 0; i < translation_errors.size(); ++i) {
            trans_rmse += translation_errors[i] * translation_errors[i];
            rot_rmse += rotation_errors[i] * rotation_errors[i];
            trans_mean += translation_errors[i];
            rot_mean += rotation_errors[i];
            if (translation_errors[i] > trans_max) trans_max = translation_errors[i];
            if (rotation_errors[i] > rot_max) rot_max = rotation_errors[i];
        }
        size_t n = translation_errors.size();
        trans_rmse = std::sqrt(trans_rmse / n);
        rot_rmse = std::sqrt(rot_rmse / n);
        trans_mean /= n;
        rot_mean /= n;
        std::cout << "\nPose estimation error (correct loop matches, " << (is_kitti ? "KITTI coords" : "sensor coords") << "):" << std::endl;
        std::cout << "  Translation RMSE: " << trans_rmse << " m, mean: " << trans_mean << " m, max: " << trans_max << " m" << std::endl;
        std::cout << "  Rotation RMSE: " << rot_rmse << " deg, mean: " << rot_mean << " deg, max: " << rot_max << " deg" << std::endl;
    }

    // 保存结果与耗时统计
    saveResults(results_file, poses, loop_pairs, match_scores, loop_is_correct_flags);
    std::cout << "\nResults saved to: " << results_file << std::endl;

    saveFrameTimingCsv(desc_timing_file, desc_timing_records);
    savePairTimingCsv(match_timing_file, match_timing_records);
    savePairTimingCsv(pose_timing_file, pose_timing_records);
    std::cout << "Descriptor timing (matched query frames): " << desc_timing_file
              << " (" << desc_timing_records.size() << " rows)" << std::endl;
    std::cout << "Match timing (successful pairs): " << match_timing_file
              << " (" << match_timing_records.size() << " rows)" << std::endl;
    std::cout << "Pose timing (successful pairs): " << pose_timing_file
              << " (" << pose_timing_records.size() << " rows)" << std::endl;

    // 保存详细的匹配信息（包含所有符合筛选条件的匹配，用于可视化）
    std::ofstream detail_out(detail_file);
    detail_out << "# Detailed match results (Semantic Triangular Descriptor)" << std::endl;
    detail_out << "# Format: current_frame_id candidate_frame_id match_score pose_distance" << std::endl;
    detail_out << "# All matches that passed distance and frame interval filtering are included" << std::endl;
    int total_filtered_matches = 0;
    for (const auto& kv : all_filtered_matches) {
        int current_frame_id = kv.first;
        for (const auto& match_result : kv.second) {
            detail_out << current_frame_id << " " 
                      << match_result.candidate_frame_id << " " 
                      << match_result.match_score << " " 
                      << match_result.pose_distance << std::endl;
            total_filtered_matches++;
        }
    }
    
    // 报告异常帧（标签异常的帧）
    detail_out << std::endl;
    detail_out << "# Anomalous frames (skipped due to high ratio of unlabeled/outlier/other-object labels)" << std::endl;
    detail_out << "# Format: frame_id" << std::endl;
    if (anomalous_frame_ids.empty()) {
        detail_out << "# No anomalous frames detected" << std::endl;
    } else {
        for (int anomalous_frame_id : anomalous_frame_ids) {
            detail_out << anomalous_frame_id << std::endl;
        }
    }
    detail_out.close();
    std::cout << "Detailed match results saved to: " << detail_file << std::endl;
    std::cout << "  Total filtered matches recorded: " << total_filtered_matches 
              << " (from " << all_filtered_matches.size() << " frames)" << std::endl;
    std::cout << "  Anomalous frames reported: " << anomalous_frame_ids.size() << std::endl;

    return 0;
}

