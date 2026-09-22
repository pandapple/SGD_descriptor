#include "include/sgd.h"
#include <algorithm>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <unordered_set>

void load_config_setting(std::string &config_file,
                         ConfigSetting &config_setting) {
  cv::FileStorage fSettings(config_file, cv::FileStorage::READ);
  if (!fSettings.isOpened()) {
    std::cerr << "Failed to open settings file at: " << config_file
              << std::endl;
    exit(-1);
  }

  // for binary descriptor
  config_setting.useful_corner_num_ = fSettings["useful_corner_num"];
  config_setting.plane_merge_normal_thre_ =
      fSettings["plane_merge_normal_thre"];
  config_setting.plane_merge_dis_thre_ = fSettings["plane_merge_dis_thre"];
  config_setting.plane_detection_thre_ = fSettings["plane_detection_thre"];
  config_setting.voxel_size_ = fSettings["voxel_size"];
  config_setting.voxel_init_num_ = fSettings["voxel_init_num"];
  config_setting.proj_plane_num_ = fSettings["proj_plane_num"];
  config_setting.proj_image_resolution_ = fSettings["proj_image_resolution"];
  config_setting.proj_image_high_inc_ = fSettings["proj_image_high_inc"];
  config_setting.proj_dis_min_ = fSettings["proj_dis_min"];
  config_setting.proj_dis_max_ = fSettings["proj_dis_max"];
  config_setting.summary_min_thre_ = fSettings["summary_min_thre"];
  config_setting.line_filter_enable_ = fSettings["line_filter_enable"];

  // std descriptor
  config_setting.descriptor_near_num_ = fSettings["descriptor_near_num"];
  if (!fSettings["descriptor_search_radius"].empty()) {
    config_setting.descriptor_search_radius_ = fSettings["descriptor_search_radius"];
  }
  config_setting.descriptor_min_len_ = fSettings["descriptor_min_len"];
  config_setting.descriptor_max_len_ = fSettings["descriptor_max_len"];
  // Support both "non_max_suppression_radius" and "max_constrait_dis" for backward compatibility
  if (fSettings["non_max_suppression_radius"].empty()) {
  config_setting.non_max_suppression_radius_ = fSettings["max_constrait_dis"];
  } else {
    config_setting.non_max_suppression_radius_ = fSettings["non_max_suppression_radius"];
  }
  if (!fSettings["triangle_resolution"].empty()) {
    config_setting.std_side_resolution_ = fSettings["triangle_resolution"];
  } else if (!fSettings["std_side_resolution"].empty()) {
    config_setting.std_side_resolution_ = fSettings["std_side_resolution"];
  }

  // candidate search
  config_setting.skip_near_num_ = fSettings["skip_near_num"];
  config_setting.candidate_num_ = fSettings["candidate_num"];
  config_setting.rough_dis_threshold_ = fSettings["rough_dis_threshold"];
  config_setting.icp_threshold_ = fSettings["icp_threshold"];
  config_setting.normal_threshold_ = fSettings["normal_threshold"];
  config_setting.dis_threshold_ = fSettings["dis_threshold"];

  // semantic matching parameters
  if (!fSettings["semantic_vertex_match_threshold"].empty()) {
    config_setting.semantic_vertex_match_threshold_ = fSettings["semantic_vertex_match_threshold"];
  }
  if (!fSettings["semantic_ratio_threshold"].empty()) {
    config_setting.semantic_ratio_threshold_ = fSettings["semantic_ratio_threshold"];
  }
  if (!fSettings["plane_node_sequence_similarity_threshold"].empty()) {
    config_setting.plane_node_sequence_similarity_threshold_ = fSettings["plane_node_sequence_similarity_threshold"];
  }
  if (!fSettings["semantic_icp_weight"].empty()) {
    config_setting.semantic_icp_weight_ = fSettings["semantic_icp_weight"];
  }
  if (!fSettings["instance_min_voxels"].empty()) {
    config_setting.instance_min_voxels_ = fSettings["instance_min_voxels"];
  }
  if (!fSettings["instance_connectivity"].empty()) {
    config_setting.instance_connectivity_ = fSettings["instance_connectivity"];
  }
  if (!fSettings["ablation_node_mode"].empty()) {
    config_setting.ablation_node_mode_ = fSettings["ablation_node_mode"];
  }
  if (!fSettings["temporal_consistency_frames"].empty()) {
    config_setting.temporal_consistency_frames_ = fSettings["temporal_consistency_frames"];
  }
  if (!fSettings["plane_vertex_weight"].empty()) {
    config_setting.plane_vertex_weight_ = fSettings["plane_vertex_weight"];
  }
  if (!fSettings["instance_vertex_weight"].empty()) {
    config_setting.instance_vertex_weight_ = fSettings["instance_vertex_weight"];
  }
  if (!fSettings["excluded_labels_non_plane"].empty()) {
    config_setting.excluded_labels_non_plane.clear();
    cv::FileNode node = fSettings["excluded_labels_non_plane"];
    if (node.isSeq()) {
      for (cv::FileNodeIterator it = node.begin(); it != node.end(); ++it) {
        config_setting.excluded_labels_non_plane.insert(
            static_cast<uint32_t>(static_cast<int>(*it)));
      }
    }
  }

  std::cout << "Sucessfully load config file:" << config_file << std::endl;
}

void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZI> &pl_feat,
                         double voxel_size) {
  int intensity = rand() % 255;
  if (voxel_size < 0.01) {
    return;
  }
  std::unordered_map<VOXEL_LOC, M_POINT> voxel_map;
  uint plsize = pl_feat.size();

  for (uint i = 0; i < plsize; i++) {
    pcl::PointXYZI &p_c = pl_feat[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      iter->second.xyz[0] += p_c.x;
      iter->second.xyz[1] += p_c.y;
      iter->second.xyz[2] += p_c.z;
      iter->second.intensity += p_c.intensity;
      iter->second.count++;
    } else {
      M_POINT anp;
      anp.xyz[0] = p_c.x;
      anp.xyz[1] = p_c.y;
      anp.xyz[2] = p_c.z;
      anp.intensity = p_c.intensity;
      anp.count = 1;
      voxel_map[position] = anp;
    }
  }
  plsize = voxel_map.size();
  pl_feat.clear();
  pl_feat.resize(plsize);

  uint i = 0;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter) {
    pl_feat[i].x = iter->second.xyz[0] / iter->second.count;
    pl_feat[i].y = iter->second.xyz[1] / iter->second.count;
    pl_feat[i].z = iter->second.xyz[2] / iter->second.count;
    pl_feat[i].intensity = iter->second.intensity / iter->second.count;
    i++;
  }
}
// 已废弃：binary 相似度不再用于匹配（similarity_threshold=0 最优），保留接口兼容
double binary_similarity(const BinaryDescriptor &b1,
                         const BinaryDescriptor &b2) {
  (void)b1;
  (void)b2;
  return 1.0;
}

bool binary_greater_sort(BinaryDescriptor a, BinaryDescriptor b) {
  return (a.summary_ > b.summary_);
}

bool plane_greater_sort(std::shared_ptr<Plane> plane1,
                        std::shared_ptr<Plane> plane2) {
  return plane1->points_size_ > plane2->points_size_;
}

void OctoTree::init_octo_tree() {
  if (voxel_points_.size() > config_setting_.voxel_init_num_) {
    init_plane();
  }
}

void OctoTree::init_plane() {
  plane_ptr_->covariance_ = Eigen::Matrix3d::Zero();
  plane_ptr_->center_ = Eigen::Vector3d::Zero();
  plane_ptr_->normal_ = Eigen::Vector3d::Zero();
  plane_ptr_->points_size_ = voxel_points_.size();
  plane_ptr_->radius_ = 0;
  for (auto pi : voxel_points_) {
    plane_ptr_->covariance_ += pi * pi.transpose();
    plane_ptr_->center_ += pi;
  }
  plane_ptr_->center_ = plane_ptr_->center_ / plane_ptr_->points_size_;
  plane_ptr_->covariance_ =
      plane_ptr_->covariance_ / plane_ptr_->points_size_ -
      plane_ptr_->center_ * plane_ptr_->center_.transpose();
  Eigen::EigenSolver<Eigen::Matrix3d> es(plane_ptr_->covariance_);
  Eigen::Matrix3cd evecs = es.eigenvectors();
  Eigen::Vector3cd evals = es.eigenvalues();
  Eigen::Vector3d evalsReal;
  evalsReal = evals.real();
  Eigen::Matrix3d::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);
  int evalsMid = 3 - evalsMin - evalsMax;
  if (evalsReal(evalsMin) < config_setting_.plane_detection_thre_) {
    plane_ptr_->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
        evecs.real()(2, evalsMin);
    plane_ptr_->min_eigen_value_ = evalsReal(evalsMin);
    plane_ptr_->radius_ = sqrt(evalsReal(evalsMax));
    plane_ptr_->is_plane_ = true;

    plane_ptr_->d_ = -(plane_ptr_->normal_(0) * plane_ptr_->center_(0) +
                       plane_ptr_->normal_(1) * plane_ptr_->center_(1) +
                       plane_ptr_->normal_(2) * plane_ptr_->center_(2));
    plane_ptr_->p_center_.x = plane_ptr_->center_(0);
    plane_ptr_->p_center_.y = plane_ptr_->center_(1);
    plane_ptr_->p_center_.z = plane_ptr_->center_(2);
    plane_ptr_->p_center_.normal_x = plane_ptr_->normal_(0);
    plane_ptr_->p_center_.normal_y = plane_ptr_->normal_(1);
    plane_ptr_->p_center_.normal_z = plane_ptr_->normal_(2);
  } else {
    plane_ptr_->is_plane_ = false;
  }
}

void publish_binary(const std::vector<BinaryDescriptor> &binary_list,
                    const Eigen::Vector3d &text_color,
                    const std::string &text_ns,
                    const ros::Publisher &text_publisher) {
  visualization_msgs::MarkerArray text_array;
  visualization_msgs::Marker text;
  text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  text.action = visualization_msgs::Marker::ADD;
  text.ns = text_ns;
  text.color.a = 0.8;  // Don't forget to set the alpha!
  text.scale.z = 0.08;
  text.pose.orientation.w = 1.0;
  text.header.frame_id = "camera_init";
  for (size_t i = 0; i < binary_list.size(); i++) {
    text.pose.position.x = binary_list[i].location_[0];
    text.pose.position.y = binary_list[i].location_[1];
    text.pose.position.z = binary_list[i].location_[2];
    std::ostringstream str;
    str << std::to_string((int)(binary_list[i].summary_));
    text.text = str.str();
    text.scale.x = 0.5;
    text.scale.y = 0.5;
    text.scale.z = 0.5;
    text.color.r = text_color[0];
    text.color.g = text_color[1];
    text.color.b = text_color[2];
    text.color.a = 1;
    text.id++;
    text_array.markers.push_back(text);
  }
  for (int i = 1; i < 100; i++) {
    text.color.a = 0;
    text.id++;
    text_array.markers.push_back(text);
  }
  text_publisher.publish(text_array);
  return;
}

void publish_std_list(const std::vector<BTC> &btc_list,
                      const ros::Publisher &std_publisher) {
  // publish descriptor
  visualization_msgs::MarkerArray ma_line;
  visualization_msgs::Marker m_line;
  m_line.type = visualization_msgs::Marker::LINE_LIST;
  m_line.action = visualization_msgs::Marker::ADD;
  m_line.ns = "std";
  // Don't forget to set the alpha!
  m_line.scale.x = 0.5;
  m_line.pose.orientation.w = 1.0;
  m_line.header.frame_id = "camera_init";
  m_line.id = 0;
  m_line.points.clear();
  m_line.color.r = 0;
  m_line.color.g = 1;
  m_line.color.b = 0;
  m_line.color.a = 1;
  for (auto var : btc_list) {
    geometry_msgs::Point p;
    p.x = var.binary_A_.location_[0];
    p.y = var.binary_A_.location_[1];
    p.z = var.binary_A_.location_[2];
    m_line.points.push_back(p);
    p.x = var.binary_B_.location_[0];
    p.y = var.binary_B_.location_[1];
    p.z = var.binary_B_.location_[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    p.x = var.binary_C_.location_[0];
    p.y = var.binary_C_.location_[1];
    p.z = var.binary_C_.location_[2];
    m_line.points.push_back(p);
    p.x = var.binary_B_.location_[0];
    p.y = var.binary_B_.location_[1];
    p.z = var.binary_B_.location_[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    p.x = var.binary_C_.location_[0];
    p.y = var.binary_C_.location_[1];
    p.z = var.binary_C_.location_[2];
    m_line.points.push_back(p);
    p.x = var.binary_A_.location_[0];
    p.y = var.binary_A_.location_[1];
    p.z = var.binary_A_.location_[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
  }
  for (int j = 0; j < 1000 * 3; j++) {
    m_line.color.a = 0.00;
    ma_line.markers.push_back(m_line);
    m_line.id++;
  }
  std_publisher.publish(ma_line);
  m_line.id = 0;
  ma_line.markers.clear();
}

void publish_std(const std::vector<std::pair<BTC, BTC>> &match_std_list,
                 const Eigen::Matrix4d &transform1,
                 const Eigen::Matrix4d &transform2,
                 const ros::Publisher &std_publisher) {
  // publish descriptor
  // bool transform_enable = true;
  visualization_msgs::MarkerArray ma_line;
  visualization_msgs::Marker m_line;
  m_line.type = visualization_msgs::Marker::LINE_LIST;
  m_line.action = visualization_msgs::Marker::ADD;
  m_line.ns = "lines";
  // Don't forget to set the alpha!
  m_line.scale.x = 0.25;
  m_line.pose.orientation.w = 1.0;
  m_line.header.frame_id = "camera_init";
  m_line.id = 0;
  int max_pub_cnt = 1;
  for (auto var : match_std_list) {
    if (max_pub_cnt > 100) {
      break;
    }
    max_pub_cnt++;
    m_line.color.a = 0.8;
    m_line.points.clear();
    // m_line.color.r = 0 / 255;
    // m_line.color.g = 233.0 / 255;
    // m_line.color.b = 0 / 255;
    m_line.color.r = 252.0 / 255;
    m_line.color.g = 233.0 / 255;
    m_line.color.b = 79.0 / 255;
    geometry_msgs::Point p;
    Eigen::Vector3d t_p;
    t_p = var.second.binary_A_.location_;
    t_p = transform2.block<3, 3>(0, 0) * t_p + transform2.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);

    t_p = var.second.binary_B_.location_;
    t_p = transform2.block<3, 3>(0, 0) * t_p + transform2.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();

    t_p = var.second.binary_C_.location_;
    t_p = transform2.block<3, 3>(0, 0) * t_p + transform2.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);

    t_p = var.second.binary_B_.location_;
    t_p = transform2.block<3, 3>(0, 0) * t_p + transform2.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();

    t_p = var.second.binary_C_.location_;
    t_p = transform2.block<3, 3>(0, 0) * t_p + transform2.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);

    t_p = var.second.binary_A_.location_;
    t_p = transform2.block<3, 3>(0, 0) * t_p + transform2.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    // another
    m_line.points.clear();
    // 252; 233; 79

    m_line.color.r = 1;
    m_line.color.g = 1;
    m_line.color.b = 1;
    // m_line.color.r = 252.0 / 255;
    // m_line.color.g = 233.0 / 255;
    // m_line.color.b = 79.0 / 255;
    t_p = var.first.binary_A_.location_;
    t_p = transform1.block<3, 3>(0, 0) * t_p + transform1.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);

    t_p = var.first.binary_B_.location_;
    t_p = transform1.block<3, 3>(0, 0) * t_p + transform1.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();

    t_p = var.first.binary_C_.location_;
    t_p = transform1.block<3, 3>(0, 0) * t_p + transform1.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    t_p = var.first.binary_B_.location_;
    t_p = transform1.block<3, 3>(0, 0) * t_p + transform1.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();

    t_p = var.first.binary_C_.location_;
    t_p = transform1.block<3, 3>(0, 0) * t_p + transform1.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    t_p = var.first.binary_A_.location_;
    t_p = transform1.block<3, 3>(0, 0) * t_p + transform1.block<3, 1>(0, 3);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    // debug
    // std_publisher.publish(ma_line);
    // std::cout << "var first: " << var.first.triangle_.transpose()
    //           << " , var second: " << var.second.triangle_.transpose()
    //           << std::endl;
    // getchar();
  }
  for (int j = 0; j < 100 * 6; j++) {
    m_line.color.a = 0.00;
    ma_line.markers.push_back(m_line);
    m_line.id++;
  }
  std_publisher.publish(ma_line);
  m_line.id = 0;
  ma_line.markers.clear();
}

double calc_triangle_dis(
    const std::vector<std::pair<BTC, BTC>> &match_std_list) {
  double mean_triangle_dis = 0;
  for (auto var : match_std_list) {
    mean_triangle_dis += (var.first.triangle_ - var.second.triangle_).norm() /
                         var.first.triangle_.norm();
  }
  if (match_std_list.size() > 0) {
    mean_triangle_dis = mean_triangle_dis / match_std_list.size();
  } else {
    mean_triangle_dis = -1;
  }
  return mean_triangle_dis;
}

// 已废弃：不再使用 binary 相似度
double calc_binary_similaity(
    const std::vector<std::pair<BTC, BTC>> &match_std_list) {
  (void)match_std_list;
  return 1.0;
}

void CalcQuation(const Eigen::Vector3d &vec, const int axis,
                 geometry_msgs::Quaternion &q) {
  Eigen::Vector3d x_body = vec;
  Eigen::Vector3d y_body(1, 1, 0);
  if (x_body(2) != 0) {
    y_body(2) = -(y_body(0) * x_body(0) + y_body(1) * x_body(1)) / x_body(2);
  } else {
    if (x_body(1) != 0) {
      y_body(1) = -(y_body(0) * x_body(0)) / x_body(1);
    } else {
      y_body(0) = 0;
    }
  }
  y_body.normalize();
  Eigen::Vector3d z_body = x_body.cross(y_body);
  Eigen::Matrix3d rot;

  rot << x_body(0), x_body(1), x_body(2), y_body(0), y_body(1), y_body(2),
      z_body(0), z_body(1), z_body(2);
  Eigen::Matrix3d rotation = rot.transpose();
  if (axis == 2) {
    Eigen::Matrix3d rot_inc;
    rot_inc << 0, 0, 1, 0, 1, 0, -1, 0, 0;
    rotation = rotation * rot_inc;
  }
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

void pubPlane(const ros::Publisher &plane_pub, const std::string plane_ns,
              const int plane_id, const pcl::PointXYZINormal normal_p,
              const float radius, const Eigen::Vector3d rgb) {
  visualization_msgs::Marker plane;
  plane.header.frame_id = "camera_init";
  plane.header.stamp = ros::Time();
  plane.ns = plane_ns;
  plane.id = plane_id;
  plane.type = visualization_msgs::Marker::CUBE;
  plane.action = visualization_msgs::Marker::ADD;
  plane.pose.position.x = normal_p.x;
  plane.pose.position.y = normal_p.y;
  plane.pose.position.z = normal_p.z;
  geometry_msgs::Quaternion q;
  Eigen::Vector3d normal_vec(normal_p.normal_x, normal_p.normal_y,
                             normal_p.normal_z);
  CalcQuation(normal_vec, 2, q);
  plane.pose.orientation = q;
  plane.scale.x = 3.0 * radius;
  plane.scale.y = 3.0 * radius;
  plane.scale.z = 0.1;
  plane.color.a = 0.8;  // 0.8
  plane.color.r = fabs(rgb(0));
  plane.color.g = fabs(rgb(1));
  plane.color.b = fabs(rgb(2));
  plane.lifetime = ros::Duration();
  plane_pub.publish(plane);
}

void SemanticTriangularDescManager::GenerateSemanticTriangularDescs(
    const pcl::PointCloud<pcl::PointXYZL>::Ptr &input_cloud, const int frame_id,
    std::vector<SemanticTriangularDescriptor> &stds_vec) {
  GenerateSemanticTriangularDescs(input_cloud, frame_id, stds_vec, nullptr, nullptr, nullptr);
}

void SemanticTriangularDescManager::GenerateSemanticTriangularDescs(
    const pcl::PointCloud<pcl::PointXYZL>::Ptr &input_cloud, const int frame_id,
    std::vector<SemanticTriangularDescriptor> &stds_vec,
    std::vector<std::vector<VOXEL_LOC>> *out_node_voxels,
    double* voxel_ms, double* desc_ms) {
  if (voxel_ms) *voxel_ms = 0.0;
  if (desc_ms) *desc_ms = 0.0;
  auto voxel_t0 = std::chrono::high_resolution_clock::now();
  // step1, voxelization with semantic labels, then build OctoTree for plane detection (same as original)
  // First, create semantic voxel map to preserve label information
  std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> semantic_voxel_map;
  init_voxel_map_semantic(input_cloud, semantic_voxel_map);
  
  // Then, build OctoTree structure from semantic voxel map (same as original version)
  std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;
  for (auto iter = semantic_voxel_map.begin(); iter != semantic_voxel_map.end(); iter++) {
    if (iter->second.size() < config_setting_.voxel_init_num_) {
      continue;
    }
    OctoTree *octo_tree = new OctoTree(config_setting_);
    for (const auto& point_label : iter->second) {
      octo_tree->voxel_points_.push_back(point_label.first);
    }
    voxel_map[iter->first] = octo_tree;
  }
  
  // Initialize octo trees in parallel (same as original version)
  std::vector<std::unordered_map<VOXEL_LOC, OctoTree *>::iterator> iter_list;
  std::vector<size_t> index;
  size_t i = 0;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter) {
    index.push_back(i);
    i++;
    iter_list.push_back(iter);
  }
  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i) { iter_list[i]->second->init_octo_tree(); });
  auto voxel_t1 = std::chrono::high_resolution_clock::now();
  if (voxel_ms) {
    *voxel_ms = std::chrono::duration<double, std::milli>(voxel_t1 - voxel_t0).count();
  }
  auto desc_t0 = voxel_t1;
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr plane_cloud(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  get_plane(voxel_map, plane_cloud);
  if (print_debug_info_) {
    std::cout << "[Description] planes size:" << plane_cloud->size()
              << std::endl;
  }

  plane_cloud_vec_.push_back(plane_cloud);
  
  // Save semantic voxel map for semantic-weighted ICP
  semantic_voxel_map_vec_.push_back(semantic_voxel_map);

  // step3, extraction binary descriptors with semantic labels
  // Use get_project_plane (same as original version) to get projection planes
  std::vector<std::shared_ptr<Plane>> proj_plane_list;
  std::vector<std::shared_ptr<Plane>> merge_plane_list;
  get_project_plane(voxel_map, proj_plane_list);
  if (proj_plane_list.size() == 0) {
    std::shared_ptr<Plane> single_plane(new Plane);
    single_plane->normal_ << 0, 0, 1;
    if (input_cloud->size() > 0) {
      single_plane->center_ << input_cloud->points[0].x, input_cloud->points[0].y,
          input_cloud->points[0].z;
    }
    merge_plane_list.push_back(single_plane);
  } else {
    sort(proj_plane_list.begin(), proj_plane_list.end(), plane_greater_sort);
    merge_plane(proj_plane_list, merge_plane_list);
    sort(merge_plane_list.begin(), merge_plane_list.end(), plane_greater_sort);
  }
  std::vector<BinaryDescriptor> binary_list;
  std::vector<std::vector<VOXEL_LOC>> node_voxels;
  size_t plane_count = 0, instance_count = 0;
  const int ablation = config_setting_.ablation_node_mode_;
  if (ablation != 2) {  // plane_only or both: 需要平面 node
    node_extractor_semantic(merge_plane_list, input_cloud, semantic_voxel_map, binary_list,
                            out_node_voxels ? &node_voxels : nullptr);
    plane_count = binary_list.size();
  }

  // 非平面体素按语义聚类，实例中心作为 node，与平面角点一起参与 3D 三角连接
  if (ablation != 1) {  // instance_only or both: 需要实例 node
    std::unordered_set<VOXEL_LOC> plane_voxels;
    for (auto it = voxel_map.begin(); it != voxel_map.end(); ++it) {
      if (it->second->plane_ptr_->is_plane_)
        plane_voxels.insert(it->first);
    }
    std::vector<BinaryDescriptor> instance_list;
    std::vector<std::vector<VOXEL_LOC>> instance_voxels;
    extract_instance_nodes_from_non_plane(semantic_voxel_map, plane_voxels, voxel_map, instance_list,
                                         out_node_voxels ? &instance_voxels : nullptr);
    instance_count = instance_list.size();
    for (size_t k = 0; k < instance_list.size(); k++) {
      binary_list.push_back(instance_list[k]);
      if (out_node_voxels) node_voxels.push_back(instance_voxels[k]);
    }
  }
  if (out_node_voxels) *out_node_voxels = std::move(node_voxels);

  history_binary_list_.push_back(binary_list);
  if (print_debug_info_) {
    std::cout << "[Description] binary size (plane: " << plane_count
              << ", instances: " << instance_count
              << ", total: " << binary_list.size() << ") [ablation_mode=" << ablation << "]" << std::endl;
  }

  // 检查二进制描述子是否为空
  if (binary_list.empty()) {
    if (print_debug_info_) {
      std::cerr << "[GenerateSemanticTriangularDescs] Warning: No binary descriptors extracted! "
                << "plane_cloud size: " << plane_cloud->size() 
                << ", merge_plane_list size: " << merge_plane_list.size()
                << ", input_cloud size: " << input_cloud->size() << std::endl;
    }
    stds_vec.clear();
    // Clean up OctoTree memory
    for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
      delete (iter->second);
    }
    auto desc_t1 = std::chrono::high_resolution_clock::now();
    if (desc_ms) {
      *desc_ms = std::chrono::duration<double, std::milli>(desc_t1 - desc_t0).count();
    }
    return;
  }

  // step4, generate stable semantic triangle descriptors
  stds_vec.clear();
  generate_semantic_triangular_desc(binary_list, frame_id, stds_vec);
  if (print_debug_info_) {
    std::cout << "[Description] semantic triangular descriptors size:" << stds_vec.size() << std::endl;
  }
  
  // step5, clear memory (same as original version)
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    delete (iter->second);
  }
  auto desc_t1 = std::chrono::high_resolution_clock::now();
  if (desc_ms) {
    *desc_ms = std::chrono::duration<double, std::milli>(desc_t1 - desc_t0).count();
  }
  return;
}

void BtcDescManager::GenerateBtcDescs(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud, const int frame_id,
    std::vector<BTC> &btcs_vec) {  // step1, voxelization and plane dection
  std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;
  init_voxel_map(input_cloud, voxel_map);
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr plane_cloud(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  get_plane(voxel_map, plane_cloud);
  if (print_debug_info_) {
    std::cout << "[Description] planes size:" << plane_cloud->size()
              << std::endl;
  }

  plane_cloud_vec_.push_back(plane_cloud);

  // step3, extraction binary descriptors
  std::vector<std::shared_ptr<Plane>> proj_plane_list;
  std::vector<std::shared_ptr<Plane>> merge_plane_list;
  get_project_plane(voxel_map, proj_plane_list);
  if (proj_plane_list.size() == 0) {
    std::shared_ptr<Plane> single_plane(new Plane);
    single_plane->normal_ << 0, 0, 1;
    single_plane->center_ << input_cloud->points[0].x, input_cloud->points[0].y,
        input_cloud->points[0].z;
    merge_plane_list.push_back(single_plane);
  } else {
    sort(proj_plane_list.begin(), proj_plane_list.end(), plane_greater_sort);
    merge_plane(proj_plane_list, merge_plane_list);
    sort(merge_plane_list.begin(), merge_plane_list.end(), plane_greater_sort);
  }
  std::vector<BinaryDescriptor> binary_list;
  node_extractor(merge_plane_list, input_cloud, binary_list);
  history_binary_list_.push_back(binary_list);
  // corner_cloud_vec_.push_back(corner_points);
  if (print_debug_info_) {
    std::cout << "[Description] binary size:" << binary_list.size()
              << std::endl;
  }

  // step4, generate stable triangle descriptors
  btcs_vec.clear();
  generate_btc(binary_list, frame_id, btcs_vec);
  if (print_debug_info_) {
    std::cout << "[Description] btcs size:" << btcs_vec.size() << std::endl;
  }
  // step5, clear memory
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    delete (iter->second);
  }
  return;
}

void SemanticTriangularDescManager::SearchLoop(
    const std::vector<SemanticTriangularDescriptor> &stds_vec, std::pair<int, double> &loop_result,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_transform,
    std::vector<std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor>> &loop_std_pair,
    double* selector_ms, double* verify_ms, double* pose_verify_ms, double* refine_ms) {
  if (stds_vec.size() == 0) {
    if (print_debug_info_) {
      std::cerr << "No SemanticTriangularDescriptors!" << std::endl;
    }
    loop_result = std::pair<int, double>(-1, 0);
    if (selector_ms) *selector_ms = 0.0;
    if (verify_ms) *verify_ms = 0.0;
    if (pose_verify_ms) *pose_verify_ms = 0.0;
    if (refine_ms) *refine_ms = 0.0;
    return;
  }
  // step1, select candidates, default number 50
  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<SemanticTriangularMatchList> candidate_matcher_vec;
  candidate_selector(stds_vec, candidate_matcher_vec);
  
  if (print_debug_info_) {
    std::cout << "[SearchLoop] Debug: Found " << candidate_matcher_vec.size() 
              << " candidate matches after candidate_selector." << std::endl;
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  // step2, select best candidates from rough candidates
  double best_score = 0;
  int best_candidate_id = -1;
  int triggle_candidate = -1;
  std::pair<Eigen::Vector3d, Eigen::Matrix3d> best_transform;
  std::vector<std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor>> best_sucess_match_vec;
  double total_pose_verify_ms = 0.0;
  double total_refine_ms = 0.0;
  for (size_t i = 0; i < candidate_matcher_vec.size(); i++) {
    double verify_score = -1;
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> relative_pose;
    std::vector<std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor>> sucess_match_vec;
    double cand_pose_verify_ms = 0.0;
    double cand_refine_ms = 0.0;
    candidate_verify(candidate_matcher_vec[i], verify_score, relative_pose,
                     sucess_match_vec, &cand_pose_verify_ms, &cand_refine_ms);
    total_pose_verify_ms += cand_pose_verify_ms;
    total_refine_ms += cand_refine_ms;
    if (print_debug_info_) {
      std::cout << "[Retreival] try frame:"
                << candidate_matcher_vec[i].match_id_.second << ", rough size:"
                << candidate_matcher_vec[i].match_list_.size()
                << ", score:" << verify_score << std::endl;
    }

    if (verify_score > best_score) {
      best_score = verify_score;
      best_candidate_id = candidate_matcher_vec[i].match_id_.second;
      best_transform = relative_pose;
      best_sucess_match_vec = sucess_match_vec;
      triggle_candidate = i;
    }
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  if (selector_ms) {
    *selector_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
  }
  if (pose_verify_ms) {
    *pose_verify_ms = total_pose_verify_ms;
  }
  if (refine_ms) {
    *refine_ms = total_refine_ms;
  }
  if (verify_ms) {
    *verify_ms = total_pose_verify_ms + total_refine_ms;
  }

  if (print_debug_info_) {
    std::cout << "[Retreival] best candidate:" << best_candidate_id
              << ", score:" << best_score 
              << ", icp_threshold:" << config_setting_.icp_threshold_ << std::endl;
  }

  // 时序一致性：连续多帧命中同一候选时可放宽阈值（参考 PNE-SGAN 等）
  bool temporal_accept = false;
  if (config_setting_.temporal_consistency_frames_ >= 2 &&
      best_candidate_id >= 0 && best_score > 0 &&
      best_candidate_id == last_best_candidate_id_ && last_best_score_ > 0) {
    double avg_score = (best_score + last_best_score_) * 0.5;
    if (avg_score > config_setting_.icp_threshold_ * 0.85f) {
      temporal_accept = true;
      if (print_debug_info_) {
        std::cout << "[SearchLoop] Temporal consistency: accept candidate " << best_candidate_id
                  << " (avg_score=" << avg_score << ")" << std::endl;
      }
    }
  }
  if (best_candidate_id >= 0) {
    last_best_candidate_id_ = best_candidate_id;
    last_best_score_ = best_score;
  } else {
    last_best_candidate_id_ = -1;
    last_best_score_ = 0.0;
  }

  if (best_score > config_setting_.icp_threshold_ || temporal_accept) {
    loop_result = std::pair<int, double>(best_candidate_id, best_score);
    loop_transform = best_transform;
    loop_std_pair = best_sucess_match_vec;
    return;
  }
  // 即使低于阈值，也返回实际得分（用于评估）
  if (best_candidate_id >= 0 && best_score > 0) {
    loop_result = std::pair<int, double>(best_candidate_id, best_score);
    loop_transform = best_transform;
    loop_std_pair = best_sucess_match_vec;
    if (print_debug_info_) {
      std::cerr << "[SearchLoop] Warning: Best score " << best_score 
                << " is below threshold " << config_setting_.icp_threshold_ 
                << ", but returning score for evaluation." << std::endl;
    }
    return;
  }
  loop_result = std::pair<int, double>(-1, 0);
  if (print_debug_info_) {
    std::cerr << "[SearchLoop] Warning: No candidate found or all candidates failed verification. "
              << "best_candidate_id=" << best_candidate_id 
              << ", best_score=" << best_score << std::endl;
  }
}

// Note: BtcDescManager::SearchLoop with BTC type is the same as SemanticTriangularDescManager::SearchLoop
// since BTC is a typedef of SemanticTriangularDescriptor. The implementation above handles both cases.

void SemanticTriangularDescManager::AddSemanticTriangularDescs(const std::vector<SemanticTriangularDescriptor> &stds_vec) {
  // update frame id
  for (auto single_std : stds_vec) {
    // calculate the position of single std
    SemanticTriangularDescriptor_LOC position;
    position.x = (int)(single_std.triangle_[0] + 0.5);
    position.y = (int)(single_std.triangle_[1] + 0.5);
    position.z = (int)(single_std.triangle_[2] + 0.5);
    auto iter = data_base_.find(position);
    if (iter != data_base_.end()) {
      data_base_[position].push_back(single_std);
    } else {
      std::vector<SemanticTriangularDescriptor> descriptor_vec;
      descriptor_vec.push_back(single_std);
      data_base_[position] = descriptor_vec;
    }
  }
  return;
}

// Note: BtcDescManager::AddBtcDescs with BTC type is the same as SemanticTriangularDescManager::AddSemanticTriangularDescs
// since BTC is a typedef of SemanticTriangularDescriptor. The implementation above handles both cases.

void BtcDescManager::PlaneGeomrtricIcp(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform,
    const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> *source_semantic_map,
    const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> *target_semantic_map) {
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  // 检查目标点云是否为空
  if (target_cloud->empty()) {
    if (print_debug_info_) {
      std::cerr << "[PlaneGeomrtricIcp] Warning: target_cloud is empty!" << std::endl;
    }
    return;
  }
  
  for (size_t i = 0; i < target_cloud->size(); i++) {
    pcl::PointXYZ pi;
    pi.x = target_cloud->points[i].x;
    pi.y = target_cloud->points[i].y;
    pi.z = target_cloud->points[i].z;
    input_cloud->push_back(pi);
  }
  kd_tree->setInputCloud(input_cloud);
  ceres::Manifold *quaternion_manifold = new ceres::EigenQuaternionManifold;
  ceres::Problem problem;
  ceres::LossFunction *loss_function = nullptr;
  Eigen::Matrix3d rot = transform.second;
  Eigen::Quaterniond q(rot);
  Eigen::Vector3d t = transform.first;
  double para_q[4] = {q.x(), q.y(), q.z(), q.w()};
  double para_t[3] = {t(0), t(1), t(2)};
  problem.AddParameterBlock(para_q, 4, quaternion_manifold);
  problem.AddParameterBlock(para_t, 3);
  Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
  Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  int useful_match = 0;
  int semantic_match_count = 0;
  
  // 检查是否启用语义权重ICP
  // 只有当 semantic_vertex_match_threshold_ > 0 时才使用语义权重
  // 如果阈值为0，则完全禁用语义功能（向后兼容模式）
  bool use_semantic_weight = (source_semantic_map != nullptr && 
                              target_semantic_map != nullptr && 
                              config_setting_.semantic_icp_weight_ > 1.0 &&
                              config_setting_.semantic_vertex_match_threshold_ > 0);
  
  for (size_t i = 0; i < source_cloud->size(); i++) {
    pcl::PointXYZINormal searchPoint = source_cloud->points[i];
    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
    pi = rot * pi + t;
    pcl::PointXYZ use_search_point;
    use_search_point.x = pi[0];
    use_search_point.y = pi[1];
    use_search_point.z = pi[2];
    Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y,
                       searchPoint.normal_z);
    ni = rot * ni;
    if (kd_tree->nearestKSearch(use_search_point, 1, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      pcl::PointXYZINormal nearstPoint =
          target_cloud->points[pointIdxNKNSearch[0]];
      Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
      Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y,
                          nearstPoint.normal_z);
      Eigen::Vector3d normal_inc = ni - tni;
      Eigen::Vector3d normal_add = ni + tni;
      double point_to_point_dis = (pi - tpi).norm();
      double point_to_plane = fabs(tni.transpose() * (pi - tpi));
      if ((normal_inc.norm() < config_setting_.normal_threshold_ ||
           normal_add.norm() < config_setting_.normal_threshold_) &&
          point_to_plane < config_setting_.dis_threshold_ &&
          point_to_point_dis < 3) {
        useful_match++;
        
        // 检查语义标签是否匹配（与candidate_selector的逻辑保持一致）
        double weight = 1.0;
        if (use_semantic_weight) {
          // 获取源点和目标点的语义标签（第一和第二标签及其比例）
          Eigen::Vector3d source_point(searchPoint.x, searchPoint.y, searchPoint.z);
          Eigen::Vector3d target_point(nearstPoint.x, nearstPoint.y, nearstPoint.z);
          
          uint32_t src_label_1, src_label_2, tgt_label_1, tgt_label_2;
          double src_ratio_1, src_ratio_2, tgt_ratio_1, tgt_ratio_2;
          
          this->get_semantic_labels_at_location(source_point, *source_semantic_map, 
                                                 src_label_1, src_ratio_1, src_label_2, src_ratio_2);
          this->get_semantic_labels_at_location(target_point, *target_semantic_map,
                                                 tgt_label_1, tgt_ratio_1, tgt_label_2, tgt_ratio_2);
          
          // 检查4种可能的匹配组合（与candidate_selector的逻辑一致）
          bool semantic_match = false;
          
          // 1. src第一标签 == tgt第一标签
          if (src_label_1 != 0 && tgt_label_1 != 0 && src_label_1 == tgt_label_1) {
            double ratio_diff = std::abs(src_ratio_1 - tgt_ratio_1);
            if (ratio_diff < config_setting_.semantic_ratio_threshold_) {
              semantic_match = true;
            }
          }
          // 2. src第一标签 == tgt第二标签
          if (!semantic_match && src_label_1 != 0 && tgt_label_2 != 0 && src_label_1 == tgt_label_2) {
            double ratio_diff = std::abs(src_ratio_1 - tgt_ratio_2);
            if (ratio_diff < config_setting_.semantic_ratio_threshold_) {
              semantic_match = true;
            }
          }
          // 3. src第二标签 == tgt第一标签
          if (!semantic_match && src_label_2 != 0 && tgt_label_1 != 0 && src_label_2 == tgt_label_1) {
            double ratio_diff = std::abs(src_ratio_2 - tgt_ratio_1);
            if (ratio_diff < config_setting_.semantic_ratio_threshold_) {
              semantic_match = true;
            }
          }
          // 4. src第二标签 == tgt第二标签
          if (!semantic_match && src_label_2 != 0 && tgt_label_2 != 0 && src_label_2 == tgt_label_2) {
            double ratio_diff = std::abs(src_ratio_2 - tgt_ratio_2);
            if (ratio_diff < config_setting_.semantic_ratio_threshold_) {
              semantic_match = true;
            }
          }
          
          // 如果语义标签匹配，增加权重
          if (semantic_match) {
            weight = config_setting_.semantic_icp_weight_;
            semantic_match_count++;
          }
        }
        
        ceres::CostFunction *cost_function;
        Eigen::Vector3d curr_point(source_cloud->points[i].x,
                                   source_cloud->points[i].y,
                                   source_cloud->points[i].z);
        Eigen::Vector3d curr_normal(source_cloud->points[i].normal_x,
                                    source_cloud->points[i].normal_y,
                                    source_cloud->points[i].normal_z);

        cost_function = PlaneSolver::Create(curr_point, curr_normal, tpi, tni);
        
        // 使用 ScaledLoss 来应用权重
        ceres::LossFunction *weighted_loss = nullptr;
        if (weight > 1.0) {
          // 使用 ScaledLoss 来增加权重（权重越大，残差的影响越大）
          weighted_loss = new ceres::ScaledLoss(nullptr, weight, ceres::TAKE_OWNERSHIP);
        }
        
        problem.AddResidualBlock(cost_function, weighted_loss, para_q, para_t);
      }
    }
  }
  
  if (print_debug_info_ && use_semantic_weight) {
    std::cout << "[PlaneGeomrtricIcp] Semantic-weighted ICP: " 
              << semantic_match_count << " / " << useful_match 
              << " matches have matching semantic labels" << std::endl;
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  Eigen::Quaterniond q_opt(para_q[3], para_q[0], para_q[1], para_q[2]);
  rot = q_opt.toRotationMatrix();
  t << t_last_curr(0), t_last_curr(1), t_last_curr(2);
  transform.first = t;
  transform.second = rot;
}

void BtcDescManager::init_voxel_map(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map) {
  uint plsize = input_cloud->size();
  for (uint i = 0; i < plsize; i++) {
    Eigen::Vector3d p_c(input_cloud->points[i].x, input_cloud->points[i].y,
                        input_cloud->points[i].z);
    double loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c[j] / config_setting_.voxel_size_;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      voxel_map[position]->voxel_points_.push_back(p_c);
    } else {
      OctoTree *octo_tree = new OctoTree(config_setting_);
      voxel_map[position] = octo_tree;
      voxel_map[position]->voxel_points_.push_back(p_c);
    }
  }
  std::vector<std::unordered_map<VOXEL_LOC, OctoTree *>::iterator> iter_list;
  std::vector<size_t> index;
  size_t i = 0;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter) {
    index.push_back(i);
    i++;
    iter_list.push_back(iter);
    // iter->second->init_octo_tree();
  }
  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i) { iter_list[i]->second->init_octo_tree(); });
}

void BtcDescManager::get_plane(
    const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud) {
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    if (iter->second->plane_ptr_->is_plane_) {
      pcl::PointXYZINormal pi;
      pi.x = iter->second->plane_ptr_->center_[0];
      pi.y = iter->second->plane_ptr_->center_[1];
      pi.z = iter->second->plane_ptr_->center_[2];
      pi.normal_x = iter->second->plane_ptr_->normal_[0];
      pi.normal_y = iter->second->plane_ptr_->normal_[1];
      pi.normal_z = iter->second->plane_ptr_->normal_[2];
      plane_cloud->push_back(pi);
    }
  }
}

// 体素化并保留每个点及其语义；每个体素的语义由其中点云的多数投票赋予（见 get_voxel_majority_semantic / get_semantic_label_at_location）
void SemanticTriangularDescManager::init_voxel_map_semantic(
    const pcl::PointCloud<pcl::PointXYZL>::Ptr &input_cloud,
    std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map) {
  uint plsize = input_cloud->size();
  for (uint i = 0; i < plsize; i++) {
    Eigen::Vector3d p_c(input_cloud->points[i].x, input_cloud->points[i].y,
                        input_cloud->points[i].z);
    uint32_t label = input_cloud->points[i].label & 0xFFFF;
    double loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c[j] / config_setting_.voxel_size_;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    voxel_map[position].push_back(std::make_pair(p_c, label));
  }
}

void SemanticTriangularDescManager::get_plane_semantic(
    const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud,
    std::unordered_set<VOXEL_LOC> &plane_voxels) {
  plane_voxels.clear();
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    if (iter->second.size() < config_setting_.voxel_init_num_) {
      continue;
    }
    
    // 使用PCA进行平面检测
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    
    for (const auto& point_label : iter->second) {
      const auto& point = point_label.first;
      covariance += point * point.transpose();
      center += point;
    }
    
    size_t point_count = iter->second.size();
    center /= point_count;
    covariance = covariance / point_count - center * center.transpose();
    
    // 特征值分解
    Eigen::EigenSolver<Eigen::Matrix3d> es(covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal = evals.real();
    
    // 找到最小特征值的索引
    Eigen::Matrix3d::Index evalsMin;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    
    // 如果最小特征值小于阈值，认为是平面
    if (evalsReal(evalsMin) < config_setting_.plane_detection_thre_) {
      pcl::PointXYZINormal pi;
      pi.x = center[0];
      pi.y = center[1];
      pi.z = center[2];
      pi.normal_x = evecs.real()(0, evalsMin);
      pi.normal_y = evecs.real()(1, evalsMin);
      pi.normal_z = evecs.real()(2, evalsMin);
      plane_cloud->push_back(pi);
      // 标记为平面体素
      plane_voxels.insert(iter->first);
    }
  }
}

// 体素内多数投票得到主语义标签及比例（第一、第二多的标签）
void SemanticTriangularDescManager::get_voxel_majority_semantic(
    const std::vector<std::pair<Eigen::Vector3d, uint32_t>> &voxel_points,
    uint32_t &majority_label, double &majority_ratio,
    uint32_t &second_label, double &second_ratio) {
  majority_label = 0;
  majority_ratio = 0.0;
  second_label = 0;
  second_ratio = 0.0;
  if (voxel_points.empty()) return;
  std::map<uint32_t, int> label_count;
  for (const auto &p : voxel_points) {
    uint32_t l = p.second & 0xFFFF;
    label_count[l]++;
  }
  std::vector<std::pair<int, uint32_t>> by_count;
  for (const auto &kv : label_count)
    by_count.push_back({kv.second, kv.first});
  std::sort(by_count.begin(), by_count.end(),
            [](const std::pair<int, uint32_t> &a, const std::pair<int, uint32_t> &b) {
              return a.first > b.first;
            });
  int total = static_cast<int>(voxel_points.size());
  if (!by_count.empty()) {
    majority_label = by_count[0].second;
    majority_ratio = static_cast<double>(by_count[0].first) / total;
    if (by_count.size() > 1) {
      second_label = by_count[1].second;
      second_ratio = static_cast<double>(by_count[1].first) / total;
    }
  }
}

// Union-Find for clustering (by index to avoid overflow)
namespace {
struct UnionFind {
  std::vector<int> parent;
  void init(int n) {
    parent.resize(n);
    for (int i = 0; i < n; i++) parent[i] = i;
  }
  int find(int i) {
    if (parent[i] != i) parent[i] = find(parent[i]);
    return parent[i];
  }
  void unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a != b) parent[a] = b;
  }
};
}  // namespace

void SemanticTriangularDescManager::extract_instance_nodes_from_non_plane(
    const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &semantic_voxel_map,
    const std::unordered_set<VOXEL_LOC> &plane_voxels,
    const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
    std::vector<BinaryDescriptor> &instance_descriptor_list,
    std::vector<std::vector<VOXEL_LOC>> *out_instance_voxels) {
  instance_descriptor_list.clear();
  const int min_voxels = config_setting_.instance_min_voxels_;
  const bool use_26 = (config_setting_.instance_connectivity_ == 26);
  const std::unordered_set<uint32_t> *excluded =
      config_setting_.excluded_labels_non_plane.empty()
          ? &config_setting_.excluded_labels
          : &config_setting_.excluded_labels_non_plane;

  // 收集非平面体素：在 voxel_map 中且非平面，且点数足够；每个体素赋予多数投票语义
  std::vector<VOXEL_LOC> non_plane_locs;
  std::unordered_map<VOXEL_LOC, uint32_t> loc_to_majority_label;
  for (auto it = voxel_map.begin(); it != voxel_map.end(); ++it) {
    const VOXEL_LOC &loc = it->first;
    if (it->second->plane_ptr_->is_plane_) continue;
    auto sm = semantic_voxel_map.find(loc);
    if (sm == semantic_voxel_map.end() || sm->second.size() < static_cast<size_t>(config_setting_.voxel_init_num_))
      continue;
    uint32_t maj_lab;
    double maj_ratio;
    uint32_t sec_lab;
    double sec_ratio;
    get_voxel_majority_semantic(sm->second, maj_lab, maj_ratio, sec_lab, sec_ratio);
    if (excluded->count(maj_lab)) continue;
    non_plane_locs.push_back(loc);
    loc_to_majority_label[loc] = maj_lab;
  }

  if (non_plane_locs.empty()) return;

  std::unordered_map<VOXEL_LOC, int> loc_to_idx;
  for (size_t i = 0; i < non_plane_locs.size(); i++)
    loc_to_idx[non_plane_locs[i]] = static_cast<int>(i);

  UnionFind uf;
  uf.init(static_cast<int>(non_plane_locs.size()));

  for (size_t i = 0; i < non_plane_locs.size(); i++) {
    const VOXEL_LOC &loc = non_plane_locs[i];
    uint32_t label0 = loc_to_majority_label[loc];
    for (int dx = -1; dx <= 1; dx++) {
      for (int dy = -1; dy <= 1; dy++) {
        for (int dz = -1; dz <= 1; dz++) {
          if (dx == 0 && dy == 0 && dz == 0) continue;
          if (!use_26 && (std::abs(dx) + std::abs(dy) + std::abs(dz)) > 1) continue;
          VOXEL_LOC nb(loc.x + dx, loc.y + dy, loc.z + dz);
          auto it = loc_to_idx.find(nb);
          if (it == loc_to_idx.end()) continue;
          if (loc_to_majority_label[nb] != label0) continue;
          uf.unite(static_cast<int>(i), it->second);
        }
      }
    }
  }

  std::unordered_map<int, std::vector<VOXEL_LOC>> root_to_locs;
  for (size_t i = 0; i < non_plane_locs.size(); i++) {
    int r = uf.find(static_cast<int>(i));
    root_to_locs[r].push_back(non_plane_locs[i]);
  }

  for (auto &kv : root_to_locs) {
    if (static_cast<int>(kv.second.size()) < min_voxels) continue;
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    int pt_count = 0;
    uint32_t instance_label = loc_to_majority_label[kv.second.front()];
    for (const VOXEL_LOC &loc : kv.second) {
      auto sm = semantic_voxel_map.find(loc);
      if (sm == semantic_voxel_map.end()) continue;
      for (const auto &p : sm->second) {
        center += p.first;
        pt_count++;
      }
    }
    if (pt_count == 0) continue;
    center /= pt_count;

    BinaryDescriptor bin;
    bin.location_ = center;
    bin.semantic_label_ = instance_label;
    bin.semantic_ratio_ = 1.0;
    bin.semantic_label_2_ = 0;
    bin.semantic_ratio_2_ = 0.0;
    bin.occupy_array_.assign(1, true);
    bin.summary_ = 1;
    bin.is_instance_node_ = true;  // 标记为实例节点，用于验证时平面/实例加权
    instance_descriptor_list.push_back(bin);
    if (out_instance_voxels) out_instance_voxels->push_back(kv.second);
  }
}

void BtcDescManager::get_project_plane(
    std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
    std::vector<std::shared_ptr<Plane>> &project_plane_list) {
  std::vector<std::shared_ptr<Plane>> origin_list;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    if (iter->second->plane_ptr_->is_plane_) {
      origin_list.push_back(iter->second->plane_ptr_);
    }
  }
  for (size_t i = 0; i < origin_list.size(); i++) origin_list[i]->id_ = 0;
  int current_id = 1;
  for (auto iter = origin_list.end() - 1; iter != origin_list.begin(); iter--) {
    for (auto iter2 = origin_list.begin(); iter2 != iter; iter2++) {
      Eigen::Vector3d normal_diff = (*iter)->normal_ - (*iter2)->normal_;
      Eigen::Vector3d normal_add = (*iter)->normal_ + (*iter2)->normal_;
      double dis1 =
          fabs((*iter)->normal_(0) * (*iter2)->center_(0) +
               (*iter)->normal_(1) * (*iter2)->center_(1) +
               (*iter)->normal_(2) * (*iter2)->center_(2) + (*iter)->d_);
      double dis2 =
          fabs((*iter2)->normal_(0) * (*iter)->center_(0) +
               (*iter2)->normal_(1) * (*iter)->center_(1) +
               (*iter2)->normal_(2) * (*iter)->center_(2) + (*iter2)->d_);
      if (normal_diff.norm() < config_setting_.plane_merge_normal_thre_ ||
          normal_add.norm() < config_setting_.plane_merge_normal_thre_)
        if (dis1 < config_setting_.plane_merge_dis_thre_ &&
            dis2 < config_setting_.plane_merge_dis_thre_) {
          if ((*iter)->id_ == 0 && (*iter2)->id_ == 0) {
            (*iter)->id_ = current_id;
            (*iter2)->id_ = current_id;
            current_id++;
          } else if ((*iter)->id_ == 0 && (*iter2)->id_ != 0)
            (*iter)->id_ = (*iter2)->id_;
          else if ((*iter)->id_ != 0 && (*iter2)->id_ == 0)
            (*iter2)->id_ = (*iter)->id_;
        }
    }
  }
  std::vector<std::shared_ptr<Plane>> merge_list;
  std::vector<int> merge_flag;

  for (size_t i = 0; i < origin_list.size(); i++) {
    auto it =
        std::find(merge_flag.begin(), merge_flag.end(), origin_list[i]->id_);
    if (it != merge_flag.end()) continue;
    if (origin_list[i]->id_ == 0) {
      continue;
    }
    std::shared_ptr<Plane> merge_plane(new Plane);
    (*merge_plane) = (*origin_list[i]);
    bool is_merge = false;
    for (size_t j = 0; j < origin_list.size(); j++) {
      if (i == j) continue;
      if (origin_list[j]->id_ == origin_list[i]->id_) {
        is_merge = true;
        Eigen::Matrix3d P_PT1 =
            (merge_plane->covariance_ +
             merge_plane->center_ * merge_plane->center_.transpose()) *
            merge_plane->points_size_;
        Eigen::Matrix3d P_PT2 =
            (origin_list[j]->covariance_ +
             origin_list[j]->center_ * origin_list[j]->center_.transpose()) *
            origin_list[j]->points_size_;
        Eigen::Vector3d merge_center =
            (merge_plane->center_ * merge_plane->points_size_ +
             origin_list[j]->center_ * origin_list[j]->points_size_) /
            (merge_plane->points_size_ + origin_list[j]->points_size_);
        Eigen::Matrix3d merge_covariance =
            (P_PT1 + P_PT2) /
                (merge_plane->points_size_ + origin_list[j]->points_size_) -
            merge_center * merge_center.transpose();
        merge_plane->covariance_ = merge_covariance;
        merge_plane->center_ = merge_center;
        merge_plane->points_size_ =
            merge_plane->points_size_ + origin_list[j]->points_size_;
        merge_plane->sub_plane_num_++;
        // for (size_t k = 0; k < origin_list[j]->cloud.size(); k++) {
        //   merge_plane->cloud.points.push_back(origin_list[j]->cloud.points[k]);
        // }
        Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance_);
        Eigen::Matrix3cd evecs = es.eigenvectors();
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal;
        evalsReal = evals.real();
        Eigen::Matrix3f::Index evalsMin, evalsMax;
        evalsReal.rowwise().sum().minCoeff(&evalsMin);
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
        merge_plane->normal_ << evecs.real()(0, evalsMin),
            evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
        merge_plane->radius_ = sqrt(evalsReal(evalsMax));
        merge_plane->d_ = -(merge_plane->normal_(0) * merge_plane->center_(0) +
                            merge_plane->normal_(1) * merge_plane->center_(1) +
                            merge_plane->normal_(2) * merge_plane->center_(2));
        merge_plane->p_center_.x = merge_plane->center_(0);
        merge_plane->p_center_.y = merge_plane->center_(1);
        merge_plane->p_center_.z = merge_plane->center_(2);
        merge_plane->p_center_.normal_x = merge_plane->normal_(0);
        merge_plane->p_center_.normal_y = merge_plane->normal_(1);
        merge_plane->p_center_.normal_z = merge_plane->normal_(2);
      }
    }
    if (is_merge) {
      merge_flag.push_back(merge_plane->id_);
      merge_list.push_back(merge_plane);
    }
  }
  project_plane_list = merge_list;
}

void BtcDescManager::merge_plane(
    std::vector<std::shared_ptr<Plane>> &origin_list,
    std::vector<std::shared_ptr<Plane>> &merge_plane_list) {
  if (origin_list.size() == 1) {
    merge_plane_list = origin_list;
    return;
  }
  for (size_t i = 0; i < origin_list.size(); i++) origin_list[i]->id_ = 0;
  int current_id = 1;
  for (auto iter = origin_list.end() - 1; iter != origin_list.begin(); iter--) {
    for (auto iter2 = origin_list.begin(); iter2 != iter; iter2++) {
      Eigen::Vector3d normal_diff = (*iter)->normal_ - (*iter2)->normal_;
      Eigen::Vector3d normal_add = (*iter)->normal_ + (*iter2)->normal_;
      double dis1 =
          fabs((*iter)->normal_(0) * (*iter2)->center_(0) +
               (*iter)->normal_(1) * (*iter2)->center_(1) +
               (*iter)->normal_(2) * (*iter2)->center_(2) + (*iter)->d_);
      double dis2 =
          fabs((*iter2)->normal_(0) * (*iter)->center_(0) +
               (*iter2)->normal_(1) * (*iter)->center_(1) +
               (*iter2)->normal_(2) * (*iter)->center_(2) + (*iter2)->d_);
      if (normal_diff.norm() < config_setting_.plane_merge_normal_thre_ ||
          normal_add.norm() < config_setting_.plane_merge_normal_thre_)
        if (dis1 < config_setting_.plane_merge_dis_thre_ &&
            dis2 < config_setting_.plane_merge_dis_thre_) {
          if ((*iter)->id_ == 0 && (*iter2)->id_ == 0) {
            (*iter)->id_ = current_id;
            (*iter2)->id_ = current_id;
            current_id++;
          } else if ((*iter)->id_ == 0 && (*iter2)->id_ != 0)
            (*iter)->id_ = (*iter2)->id_;
          else if ((*iter)->id_ != 0 && (*iter2)->id_ == 0)
            (*iter2)->id_ = (*iter)->id_;
        }
    }
  }
  std::vector<int> merge_flag;

  for (size_t i = 0; i < origin_list.size(); i++) {
    auto it =
        std::find(merge_flag.begin(), merge_flag.end(), origin_list[i]->id_);
    if (it != merge_flag.end()) continue;
    if (origin_list[i]->id_ == 0) {
      merge_plane_list.push_back(origin_list[i]);
      continue;
    }
    std::shared_ptr<Plane> merge_plane(new Plane);
    (*merge_plane) = (*origin_list[i]);
    bool is_merge = false;
    for (size_t j = 0; j < origin_list.size(); j++) {
      if (i == j) continue;
      if (origin_list[j]->id_ == origin_list[i]->id_) {
        is_merge = true;
        Eigen::Matrix3d P_PT1 =
            (merge_plane->covariance_ +
             merge_plane->center_ * merge_plane->center_.transpose()) *
            merge_plane->points_size_;
        Eigen::Matrix3d P_PT2 =
            (origin_list[j]->covariance_ +
             origin_list[j]->center_ * origin_list[j]->center_.transpose()) *
            origin_list[j]->points_size_;
        Eigen::Vector3d merge_center =
            (merge_plane->center_ * merge_plane->points_size_ +
             origin_list[j]->center_ * origin_list[j]->points_size_) /
            (merge_plane->points_size_ + origin_list[j]->points_size_);
        Eigen::Matrix3d merge_covariance =
            (P_PT1 + P_PT2) /
                (merge_plane->points_size_ + origin_list[j]->points_size_) -
            merge_center * merge_center.transpose();
        merge_plane->covariance_ = merge_covariance;
        merge_plane->center_ = merge_center;
        merge_plane->points_size_ =
            merge_plane->points_size_ + origin_list[j]->points_size_;
        merge_plane->sub_plane_num_ += origin_list[j]->sub_plane_num_;
        // for (size_t k = 0; k < origin_list[j]->cloud.size(); k++) {
        //   merge_plane->cloud.points.push_back(origin_list[j]->cloud.points[k]);
        // }
        Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance_);
        Eigen::Matrix3cd evecs = es.eigenvectors();
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal;
        evalsReal = evals.real();
        Eigen::Matrix3f::Index evalsMin, evalsMax;
        evalsReal.rowwise().sum().minCoeff(&evalsMin);
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
        merge_plane->normal_ << evecs.real()(0, evalsMin),
            evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
        merge_plane->radius_ = sqrt(evalsReal(evalsMax));
        merge_plane->d_ = -(merge_plane->normal_(0) * merge_plane->center_(0) +
                            merge_plane->normal_(1) * merge_plane->center_(1) +
                            merge_plane->normal_(2) * merge_plane->center_(2));
        merge_plane->p_center_.x = merge_plane->center_(0);
        merge_plane->p_center_.y = merge_plane->center_(1);
        merge_plane->p_center_.z = merge_plane->center_(2);
        merge_plane->p_center_.normal_x = merge_plane->normal_(0);
        merge_plane->p_center_.normal_y = merge_plane->normal_(1);
        merge_plane->p_center_.normal_z = merge_plane->normal_(2);
      }
    }
    if (is_merge) {
      merge_flag.push_back(merge_plane->id_);
      merge_plane_list.push_back(merge_plane);
    }
  }
}

void SemanticTriangularDescManager::node_extractor_semantic(
    const std::vector<std::shared_ptr<Plane>> proj_plane_list,
    const pcl::PointCloud<pcl::PointXYZL>::Ptr &input_cloud,
    const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map,
    std::vector<BinaryDescriptor> &binary_descriptor_list,
    std::vector<std::vector<VOXEL_LOC>> *out_plane_voxels) {
  binary_descriptor_list.clear();
  if (out_plane_voxels) out_plane_voxels->clear();
  std::vector<BinaryDescriptor> temp_binary_list;
  std::vector<std::vector<VOXEL_LOC>> temp_voxels;
  Eigen::Vector3d last_normal(0, 0, 0);
  int useful_proj_num = 0;
  for (int i = 0; i < proj_plane_list.size(); i++) {
    std::vector<BinaryDescriptor> prepare_binary_list;
    std::vector<std::vector<VOXEL_LOC>> prepare_voxels;
    Eigen::Vector3d proj_center = proj_plane_list[i]->center_;
    Eigen::Vector3d proj_normal = proj_plane_list[i]->normal_;
    if (proj_normal.z() < 0) {
      proj_normal = -proj_normal;
    }
    if ((proj_normal - last_normal).norm() < 0.3 ||
        (proj_normal + last_normal).norm() > 0.3) {
      last_normal = proj_normal;
      if (print_debug_info_) {
        std::cout << "[Description] reference plane normal:"
                  << proj_normal.transpose()
                  << ", center:" << proj_center.transpose() << std::endl;
      }
      useful_proj_num++;
      extract_node_semantic(proj_center, proj_normal, input_cloud, voxel_map,
                           prepare_binary_list,
                           out_plane_voxels ? &prepare_voxels : nullptr);
      for (size_t k = 0; k < prepare_binary_list.size(); k++) {
        temp_binary_list.push_back(prepare_binary_list[k]);
        if (out_plane_voxels)
          temp_voxels.push_back(prepare_voxels[k]);
      }
      if (useful_proj_num == config_setting_.proj_plane_num_) {
        break;
      }
    }
  }
  non_maxi_suppression(temp_binary_list, out_plane_voxels ? &temp_voxels : nullptr);
  if (config_setting_.useful_corner_num_ > temp_binary_list.size()) {
    binary_descriptor_list = temp_binary_list;
    if (out_plane_voxels) *out_plane_voxels = temp_voxels;
  } else {
    std::vector<size_t> idx(temp_binary_list.size());
    for (size_t i = 0; i < idx.size(); i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
      return binary_greater_sort(temp_binary_list[a], temp_binary_list[b]);
    });
    for (size_t i = 0; i < config_setting_.useful_corner_num_; i++) {
      binary_descriptor_list.push_back(temp_binary_list[idx[i]]);
      if (out_plane_voxels)
        out_plane_voxels->push_back(temp_voxels[idx[i]]);
    }
  }
  return;
}

void BtcDescManager::node_extractor(
    const std::vector<std::shared_ptr<Plane>> proj_plane_list,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::vector<BinaryDescriptor> &binary_descriptor_list) {
  binary_descriptor_list.clear();
  std::vector<BinaryDescriptor> temp_binary_list;
  Eigen::Vector3d last_normal(0, 0, 0);
  int useful_proj_num = 0;
  for (int i = 0; i < proj_plane_list.size(); i++) {
    std::vector<BinaryDescriptor> prepare_binary_list;
    Eigen::Vector3d proj_center = proj_plane_list[i]->center_;
    Eigen::Vector3d proj_normal = proj_plane_list[i]->normal_;
    if (proj_normal.z() < 0) {
      proj_normal = -proj_normal;
    }
    if ((proj_normal - last_normal).norm() < 0.3 ||
        (proj_normal + last_normal).norm() > 0.3) {
      last_normal = proj_normal;
      std::cout << "[Description] reference plane normal:"
                << proj_normal.transpose()
                << ", center:" << proj_center.transpose() << std::endl;
      useful_proj_num++;
      extract_node(proj_center, proj_normal, input_cloud,
                   prepare_binary_list);
      for (auto bi : prepare_binary_list) {
        temp_binary_list.push_back(bi);
      }
      if (useful_proj_num == config_setting_.proj_plane_num_) {
        break;
      }
    }
  }
  non_maxi_suppression(temp_binary_list);
  if (config_setting_.useful_corner_num_ > temp_binary_list.size()) {
    binary_descriptor_list = temp_binary_list;
  } else {
    std::sort(temp_binary_list.begin(), temp_binary_list.end(),
              binary_greater_sort);
    for (size_t i = 0; i < config_setting_.useful_corner_num_; i++) {
      binary_descriptor_list.push_back(temp_binary_list[i]);
    }
  }
  return;
}

void BtcDescManager::extract_node(
    const Eigen::Vector3d &project_center,
    const Eigen::Vector3d &project_normal,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::vector<BinaryDescriptor> &binary_list) {
  binary_list.clear();
  double binary_min_dis = config_setting_.summary_min_thre_;
  double resolution = config_setting_.proj_image_resolution_;
  double dis_threshold_min = config_setting_.proj_dis_min_;
  double dis_threshold_max = config_setting_.proj_dis_max_;
  double high_inc = config_setting_.proj_image_high_inc_;
  bool line_filter_enable = config_setting_.line_filter_enable_;
  double A = project_normal[0];
  double B = project_normal[1];
  double C = project_normal[2];
  double D =
      -(A * project_center[0] + B * project_center[1] + C * project_center[2]);
  std::vector<Eigen::Vector3d> projection_points;
  // Eigen::Vector3d x_axis(1, 1, 0);
  Eigen::Vector3d x_axis(1, 0, 0);
  if (C != 0) {
    x_axis[2] = -(A + B) / C;
  } else if (B != 0) {
    x_axis[1] = -A / B;
  } else {
    x_axis[0] = 0;
    x_axis[1] = 1;
  }
  x_axis.normalize();
  Eigen::Vector3d y_axis = project_normal.cross(x_axis);
  y_axis.normalize();
  double ax = x_axis[0];
  double bx = x_axis[1];
  double cx = x_axis[2];
  double dx = -(ax * project_center[0] + bx * project_center[1] +
                cx * project_center[2]);
  double ay = y_axis[0];
  double by = y_axis[1];
  double cy = y_axis[2];
  double dy = -(ay * project_center[0] + by * project_center[1] +
                cy * project_center[2]);
  std::vector<Eigen::Vector2d> point_list_2d;
  pcl::PointCloud<pcl::PointXYZ> point_list_3d;
  std::vector<double> dis_list_2d;
  for (size_t i = 0; i < input_cloud->size(); i++) {
    double x = input_cloud->points[i].x;
    double y = input_cloud->points[i].y;
    double z = input_cloud->points[i].z;
    double dis = x * A + y * B + z * C + D;
    pcl::PointXYZ pi;
    if (dis < dis_threshold_min || dis > dis_threshold_max) {
      continue;
    } else {
      if (dis > dis_threshold_min && dis <= dis_threshold_max) {
        pi.x = x;
        pi.y = y;
        pi.z = z;
      }
    }
    Eigen::Vector3d cur_project;

    cur_project[0] = (-A * (B * y + C * z + D) + x * (B * B + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[1] = (-B * (A * x + C * z + D) + y * (A * A + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[2] = (-C * (A * x + B * y + D) + z * (A * A + B * B)) /
                     (A * A + B * B + C * C);
    pcl::PointXYZ p;
    p.x = cur_project[0];
    p.y = cur_project[1];
    p.z = cur_project[2];
    double project_x =
        cur_project[0] * ay + cur_project[1] * by + cur_project[2] * cy + dy;
    double project_y =
        cur_project[0] * ax + cur_project[1] * bx + cur_project[2] * cx + dx;
    Eigen::Vector2d p_2d(project_x, project_y);
    point_list_2d.push_back(p_2d);
    dis_list_2d.push_back(dis);
    point_list_3d.points.push_back(pi);
  }
  double min_x = 10;
  double max_x = -10;
  double min_y = 10;
  double max_y = -10;
  if (point_list_2d.size() <= 5) {
    return;
  }
  for (auto pi : point_list_2d) {
    if (pi[0] < min_x) {
      min_x = pi[0];
    }
    if (pi[0] > max_x) {
      max_x = pi[0];
    }
    if (pi[1] < min_y) {
      min_y = pi[1];
    }
    if (pi[1] > max_y) {
      max_y = pi[1];
    }
  }
  // segment project cloud
  int segmen_base_num = 5;
  double segmen_len = segmen_base_num * resolution;
  int x_segment_num = (max_x - min_x) / segmen_len + 1;
  int y_segment_num = (max_y - min_y) / segmen_len + 1;
  int x_axis_len = (int)((max_x - min_x) / resolution + segmen_base_num);
  int y_axis_len = (int)((max_y - min_y) / resolution + segmen_base_num);

  std::vector<double> **dis_container = new std::vector<double> *[x_axis_len];
  BinaryDescriptor **binary_container = new BinaryDescriptor *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    dis_container[i] = new std::vector<double>[y_axis_len];
    binary_container[i] = new BinaryDescriptor[y_axis_len];
  }
  double **img_count = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    img_count[i] = new double[y_axis_len];
  }
  double **dis_array = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    dis_array[i] = new double[y_axis_len];
  }
  double **mean_x_list = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    mean_x_list[i] = new double[y_axis_len];
  }
  double **mean_y_list = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    mean_y_list[i] = new double[y_axis_len];
  }
  double **sum_x_3d = new double *[x_axis_len];
  double **sum_y_3d = new double *[x_axis_len];
  double **sum_z_3d = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    sum_x_3d[i] = new double[y_axis_len];
    sum_y_3d[i] = new double[y_axis_len];
    sum_z_3d[i] = new double[y_axis_len];
  }
  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      img_count[x][y] = 0;
      mean_x_list[x][y] = 0;
      mean_y_list[x][y] = 0;
      sum_x_3d[x][y] = 0;
      sum_y_3d[x][y] = 0;
      sum_z_3d[x][y] = 0;
      dis_array[x][y] = 0;
      std::vector<double> single_dis_container;
      dis_container[x][y] = single_dis_container;
    }
  }

  for (size_t i = 0; i < point_list_2d.size(); i++) {
    int x_index = (int)((point_list_2d[i][0] - min_x) / resolution);
    int y_index = (int)((point_list_2d[i][1] - min_y) / resolution);
    mean_x_list[x_index][y_index] += point_list_2d[i][0];
    mean_y_list[x_index][y_index] += point_list_2d[i][1];
    sum_x_3d[x_index][y_index] += point_list_3d.points[i].x;
    sum_y_3d[x_index][y_index] += point_list_3d.points[i].y;
    sum_z_3d[x_index][y_index] += point_list_3d.points[i].z;
    img_count[x_index][y_index]++;
    dis_container[x_index][y_index].push_back(dis_list_2d[i]);
  }

  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      // calc segment dis array
      if (img_count[x][y] > 0) {
        int cut_num = (dis_threshold_max - dis_threshold_min) / high_inc;
        std::vector<bool> occup_list;
        std::vector<double> cnt_list;
        BinaryDescriptor single_binary;
        for (size_t i = 0; i < cut_num; i++) {
          cnt_list.push_back(0);
          occup_list.push_back(false);
        }
        for (size_t j = 0; j < dis_container[x][y].size(); j++) {
          int cnt_index =
              (dis_container[x][y][j] - dis_threshold_min) / high_inc;
          cnt_list[cnt_index]++;
        }
        double segmnt_dis = 0;
        for (size_t i = 0; i < cut_num; i++) {
          if (cnt_list[i] >= 1) {
            segmnt_dis++;
            occup_list[i] = true;
          }
        }
        dis_array[x][y] = segmnt_dis;
        // 已废弃 binary 用于相似度匹配，不再存储 occupy_array_；summary_ 仍用于 NMS
        single_binary.occupy_array_.assign(1, true);
        single_binary.summary_ = static_cast<unsigned char>(segmnt_dis);
        binary_container[x][y] = single_binary;
      }
    }
  }

  // filter by distance
  std::vector<double> max_dis_list;
  std::vector<int> max_dis_x_index_list;
  std::vector<int> max_dis_y_index_list;

  for (int x_segment_index = 0; x_segment_index < x_segment_num;
       x_segment_index++) {
    for (int y_segment_index = 0; y_segment_index < y_segment_num;
         y_segment_index++) {
      double max_dis = 0;
      int max_dis_x_index = -10;
      int max_dis_y_index = -10;
      for (int x_index = x_segment_index * segmen_base_num;
           x_index < (x_segment_index + 1) * segmen_base_num; x_index++) {
        for (int y_index = y_segment_index * segmen_base_num;
             y_index < (y_segment_index + 1) * segmen_base_num; y_index++) {
          if (dis_array[x_index][y_index] > max_dis) {
            max_dis = dis_array[x_index][y_index];
            max_dis_x_index = x_index;
            max_dis_y_index = y_index;
          }
        }
      }
      if (max_dis >= binary_min_dis) {
        max_dis_list.push_back(max_dis);
        max_dis_x_index_list.push_back(max_dis_x_index);
        max_dis_y_index_list.push_back(max_dis_y_index);
      }
    }
  }
  // calc line or not
  std::vector<Eigen::Vector2i> direction_list;
  Eigen::Vector2i d(0, 1);
  direction_list.push_back(d);
  d << 1, 0;
  direction_list.push_back(d);
  d << 1, 1;
  direction_list.push_back(d);
  d << 1, -1;
  direction_list.push_back(d);
  for (size_t i = 0; i < max_dis_list.size(); i++) {
    Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
    if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 ||
        p[1] >= y_axis_len - 1) {
      continue;
    }
    bool is_add = true;

    if (line_filter_enable) {
      for (int j = 0; j < 4; j++) {
        Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
        if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 ||
            p[1] >= y_axis_len - 1) {
          continue;
        }
        Eigen::Vector2i p1 = p + direction_list[j];
        Eigen::Vector2i p2 = p - direction_list[j];
        double threshold = dis_array[p[0]][p[1]] - 3;
        if (dis_array[p1[0]][p1[1]] >= threshold) {
          if (dis_array[p2[0]][p2[1]] >= 0.5 * dis_array[p[0]][p[1]]) {
            is_add = false;
          }
        }
        if (dis_array[p2[0]][p2[1]] >= threshold) {
          if (dis_array[p1[0]][p1[1]] >= 0.5 * dis_array[p[0]][p[1]]) {
            is_add = false;
          }
        }
        if (dis_array[p1[0]][p1[1]] >= threshold) {
          if (dis_array[p2[0]][p2[1]] >= threshold) {
            is_add = false;
          }
        }
        if (dis_array[p2[0]][p2[1]] >= threshold) {
          if (dis_array[p1[0]][p1[1]] >= threshold) {
            is_add = false;
          }
        }
      }
    }
    if (is_add) {
      int xi = max_dis_x_index_list[i];
      int yi = max_dis_y_index_list[i];
      double cnt = img_count[xi][yi];
      // 使用网格内点的 3D 质心作为角点位置，保留真实高度，使节点和三角形在空间中分布而非共面
      Eigen::Vector3d coord(
          sum_x_3d[xi][yi] / cnt,
          sum_y_3d[xi][yi] / cnt,
          sum_z_3d[xi][yi] / cnt);
      pcl::PointXYZ pi;
      pi.x = coord[0];
      pi.y = coord[1];
      pi.z = coord[2];
      BinaryDescriptor single_binary = binary_container[xi][yi];
      single_binary.location_ = coord;
      binary_list.push_back(single_binary);
    }
  }
  for (int i = 0; i < x_axis_len; i++) {
    delete[] binary_container[i];
    delete[] dis_container[i];
    delete[] img_count[i];
    delete[] dis_array[i];
    delete[] mean_x_list[i];
    delete[] mean_y_list[i];
    delete[] sum_x_3d[i];
    delete[] sum_y_3d[i];
    delete[] sum_z_3d[i];
  }
  delete[] binary_container;
  delete[] dis_container;
  delete[] img_count;
  delete[] dis_array;
  delete[] mean_x_list;
  delete[] mean_y_list;
  delete[] sum_x_3d;
  delete[] sum_y_3d;
  delete[] sum_z_3d;
}

// 从位置获取语义标签（获取该体素中最频繁的标签）
uint32_t SemanticTriangularDescManager::get_semantic_label_at_location(
    const Eigen::Vector3d &location,
    const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map) {
  // 计算位置所在的体素位置
  double loc_xyz[3];
  loc_xyz[0] = location[0] / config_setting_.voxel_size_;
  loc_xyz[1] = location[1] / config_setting_.voxel_size_;
  loc_xyz[2] = location[2] / config_setting_.voxel_size_;
  for (int j = 0; j < 3; j++) {
    if (loc_xyz[j] < 0) {
      loc_xyz[j] -= 1.0;
    }
  }
  VOXEL_LOC voxel_pos((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  
  // 查找对应的体素
  auto iter = voxel_map.find(voxel_pos);
  if (iter != voxel_map.end() && !iter->second.empty()) {
    // 统计每个标签的数量
    std::map<uint32_t, int> label_count;
    for (const auto& point_label : iter->second) {
      uint32_t label = point_label.second & 0xFFFF;
      label_count[label]++;
    }
    
    // 返回最频繁的标签
    uint32_t most_frequent_label = 0;
    int max_count = 0;
    for (const auto& pair : label_count) {
      if (pair.second > max_count) {
        max_count = pair.second;
        most_frequent_label = pair.first;
      }
    }
    return most_frequent_label;
  }
  
  return 0;  // 默认返回0（无标签）
}

double SemanticTriangularDescManager::get_semantic_ratio(const Eigen::Vector3d &location, uint32_t semantic_label,
                            const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map) {
  // 计算位置所在的体素位置
  double loc_xyz[3];
  loc_xyz[0] = location[0] / config_setting_.voxel_size_;
  loc_xyz[1] = location[1] / config_setting_.voxel_size_;
  loc_xyz[2] = location[2] / config_setting_.voxel_size_;
  for (int j = 0; j < 3; j++) {
    if (loc_xyz[j] < 0) {
      loc_xyz[j] -= 1.0;
    }
  }
  VOXEL_LOC voxel_pos((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  
  // 查找对应的体素
  auto iter = voxel_map.find(voxel_pos);
  if (iter != voxel_map.end()) {
    // 统计该语义标签在体素中的数量
    int label_count = 0;
    int total_count = iter->second.size();
    for (const auto& point_label : iter->second) {
      uint32_t label = point_label.second & 0xFFFF;
      if (label == semantic_label) {
        label_count++;
      }
    }
    // 返回比例
    return (total_count > 0) ? static_cast<double>(label_count) / total_count : 0.0;
  }
  
  // 如果找不到对应的体素，返回0（表示比例未知）
  return 0.0;
}

// 获取位置处的第一和第二语义标签及其比例
void SemanticTriangularDescManager::get_semantic_labels_at_location(
    const Eigen::Vector3d &location,
    const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map,
    uint32_t &label_1, double &ratio_1, uint32_t &label_2, double &ratio_2) {
  // 初始化返回值
  label_1 = 0;
  ratio_1 = 0.0;
  label_2 = 0;
  ratio_2 = 0.0;
  
  // 计算位置所在的体素位置
  double loc_xyz[3];
  loc_xyz[0] = location[0] / config_setting_.voxel_size_;
  loc_xyz[1] = location[1] / config_setting_.voxel_size_;
  loc_xyz[2] = location[2] / config_setting_.voxel_size_;
  for (int j = 0; j < 3; j++) {
    if (loc_xyz[j] < 0) {
      loc_xyz[j] -= 1.0;
    }
  }
  VOXEL_LOC voxel_pos((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
  
  // 查找对应的体素
  auto iter = voxel_map.find(voxel_pos);
  if (iter != voxel_map.end() && !iter->second.empty()) {
    // 统计每个标签的数量
    std::map<uint32_t, int> label_count;
    int total_count = iter->second.size();
    for (const auto& point_label : iter->second) {
      uint32_t label = point_label.second & 0xFFFF;
      label_count[label]++;
    }
    
    // 找到频率最高和第二高的标签
    uint32_t most_frequent_label = 0;
    uint32_t second_frequent_label = 0;
    int max_count = 0;
    int second_max_count = 0;
    
    for (const auto& pair : label_count) {
      if (pair.second > max_count) {
        second_max_count = max_count;
        second_frequent_label = most_frequent_label;
        max_count = pair.second;
        most_frequent_label = pair.first;
      } else if (pair.second > second_max_count) {
        second_max_count = pair.second;
        second_frequent_label = pair.first;
      }
    }
    
    // 返回第一和第二标签及其比例
    label_1 = most_frequent_label;
    ratio_1 = (total_count > 0) ? static_cast<double>(max_count) / total_count : 0.0;
    label_2 = second_frequent_label;
    ratio_2 = (total_count > 0) ? static_cast<double>(second_max_count) / total_count : 0.0;
  }
}

void SemanticTriangularDescManager::get_plane_node_voxel_column(
    double cell_min_x, double cell_max_x, double cell_min_y, double cell_max_y,
    double A, double B, double C, double D,
    double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy,
    const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map,
    std::vector<VOXEL_LOC> &voxel_column) {
  voxel_column.clear();
  const double vs = config_setting_.voxel_size_;
  struct VoxelDist {
    VOXEL_LOC loc;
    double dist;
  };
  std::vector<VoxelDist> column;
  for (const auto &kv : voxel_map) {
    const VOXEL_LOC &loc = kv.first;
    double vcx = (loc.x + 0.5) * vs;
    double vcy = (loc.y + 0.5) * vs;
    double vcz = (loc.z + 0.5) * vs;
    Eigen::Vector3d cur_project(
        (-A * (B * vcy + C * vcz + D) + vcx * (B * B + C * C)) / (A * A + B * B + C * C),
        (-B * (A * vcx + C * vcz + D) + vcy * (A * A + C * C)) / (A * A + B * B + C * C),
        (-C * (A * vcx + B * vcy + D) + vcz * (A * A + B * B)) / (A * A + B * B + C * C));
    double project_x = cur_project[0] * ay + cur_project[1] * by + cur_project[2] * cy + dy;
    double project_y = cur_project[0] * ax + cur_project[1] * bx + cur_project[2] * cx + dx;
    if (project_x >= cell_min_x && project_x < cell_max_x &&
        project_y >= cell_min_y && project_y < cell_max_y) {
      double signed_dist = A * vcx + B * vcy + C * vcz + D;
      column.push_back({loc, signed_dist});
    }
  }
  if (column.empty()) return;
  std::sort(column.begin(), column.end(),
            [](const VoxelDist &a, const VoxelDist &b) { return a.dist < b.dist; });
  for (const auto &vd : column)
    voxel_column.push_back(vd.loc);
}

void SemanticTriangularDescManager::get_plane_node_label_sequence(
    double cell_min_x, double cell_max_x, double cell_min_y, double cell_max_y,
    double A, double B, double C, double D,
    double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy,
    const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map,
    std::vector<uint32_t> &label_sequence) {
  label_sequence.clear();
  const double vs = config_setting_.voxel_size_;
  struct VoxelDist {
    VOXEL_LOC loc;
    double dist;
  };
  std::vector<VoxelDist> column;
  for (const auto &kv : voxel_map) {
    const VOXEL_LOC &loc = kv.first;
    double cx = (loc.x + 0.5) * vs;
    double cy = (loc.y + 0.5) * vs;
    double cz = (loc.z + 0.5) * vs;
    Eigen::Vector3d cur_project(
        (-A * (B * cy + C * cz + D) + cx * (B * B + C * C)) / (A * A + B * B + C * C),
        (-B * (A * cx + C * cz + D) + cy * (A * A + C * C)) / (A * A + B * B + C * C),
        (-C * (A * cx + B * cy + D) + cz * (A * A + B * B)) / (A * A + B * B + C * C));
    double project_x = cur_project[0] * ay + cur_project[1] * by + cur_project[2] * cy + dy;
    double project_y = cur_project[0] * ax + cur_project[1] * bx + cur_project[2] * cx + dx;
    if (project_x >= cell_min_x && project_x < cell_max_x &&
        project_y >= cell_min_y && project_y < cell_max_y) {
      double signed_dist = A * cx + B * cy + C * cz + D;
      column.push_back({loc, signed_dist});
    }
  }
  if (column.empty()) return;
  std::sort(column.begin(), column.end(),
            [](const VoxelDist &a, const VoxelDist &b) { return a.dist < b.dist; });
  for (const auto &vd : column) {
    auto it = voxel_map.find(vd.loc);
    if (it == voxel_map.end() || it->second.empty()) continue;
    uint32_t maj_lab;
    double maj_ratio;
    uint32_t sec_lab;
    double sec_ratio;
    get_voxel_majority_semantic(it->second, maj_lab, maj_ratio, sec_lab, sec_ratio);
    label_sequence.push_back(maj_lab);
  }
}

void SemanticTriangularDescManager::extract_node_semantic(
    const Eigen::Vector3d &project_center,
    const Eigen::Vector3d &project_normal,
    const pcl::PointCloud<pcl::PointXYZL>::Ptr &input_cloud,
    const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> &voxel_map,
    std::vector<BinaryDescriptor> &binary_list,
    std::vector<std::vector<VOXEL_LOC>> *out_voxels) {
  // 与 extract_node 逻辑一致，并处理语义标签
  binary_list.clear();
  double binary_min_dis = config_setting_.summary_min_thre_;
  double resolution = config_setting_.proj_image_resolution_;
  double dis_threshold_min = config_setting_.proj_dis_min_;
  double dis_threshold_max = config_setting_.proj_dis_max_;
  double high_inc = config_setting_.proj_image_high_inc_;
  bool line_filter_enable = config_setting_.line_filter_enable_;
  double A = project_normal[0];
  double B = project_normal[1];
  double C = project_normal[2];
  double D =
      -(A * project_center[0] + B * project_center[1] + C * project_center[2]);
  std::vector<Eigen::Vector3d> projection_points;
  Eigen::Vector3d x_axis(1, 0, 0);
  if (C != 0) {
    x_axis[2] = -(A + B) / C;
  } else if (B != 0) {
    x_axis[1] = -A / B;
  } else {
    x_axis[0] = 0;
    x_axis[1] = 1;
  }
  x_axis.normalize();
  Eigen::Vector3d y_axis = project_normal.cross(x_axis);
  y_axis.normalize();
  double ax = x_axis[0];
  double bx = x_axis[1];
  double cx = x_axis[2];
  double dx = -(ax * project_center[0] + bx * project_center[1] +
                cx * project_center[2]);
  double ay = y_axis[0];
  double by = y_axis[1];
  double cy = y_axis[2];
  double dy = -(ay * project_center[0] + by * project_center[1] +
                cy * project_center[2]);
  std::vector<Eigen::Vector2d> point_list_2d;
  pcl::PointCloud<pcl::PointXYZ> point_list_3d;
  std::vector<double> dis_list_2d;
  std::vector<uint32_t> label_list;  // 添加标签列表
  for (size_t i = 0; i < input_cloud->size(); i++) {
    double x = input_cloud->points[i].x;
    double y = input_cloud->points[i].y;
    double z = input_cloud->points[i].z;
    double dis = x * A + y * B + z * C + D;
    pcl::PointXYZ pi;
    if (dis < dis_threshold_min || dis > dis_threshold_max) {
      continue;
    } else {
      if (dis > dis_threshold_min && dis <= dis_threshold_max) {
        pi.x = x;
        pi.y = y;
        pi.z = z;
      }
    }
    Eigen::Vector3d cur_project;

    cur_project[0] = (-A * (B * y + C * z + D) + x * (B * B + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[1] = (-B * (A * x + C * z + D) + y * (A * A + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[2] = (-C * (A * x + B * y + D) + z * (A * A + B * B)) /
                     (A * A + B * B + C * C);
    pcl::PointXYZ p;
    p.x = cur_project[0];
    p.y = cur_project[1];
    p.z = cur_project[2];
    double project_x =
        cur_project[0] * ay + cur_project[1] * by + cur_project[2] * cy + dy;
    double project_y =
        cur_project[0] * ax + cur_project[1] * bx + cur_project[2] * cx + dx;
    Eigen::Vector2d p_2d(project_x, project_y);
    point_list_2d.push_back(p_2d);
    dis_list_2d.push_back(dis);
    point_list_3d.points.push_back(pi);
    label_list.push_back(input_cloud->points[i].label & 0xFFFF);  // 保存标签
  }
  double min_x = 10;
  double max_x = -10;
  double min_y = 10;
  double max_y = -10;
  if (point_list_2d.size() <= 5) {
    if (print_debug_info_) {
      std::cerr << "[extract_node_semantic] Warning: Too few points in projection ("
                << point_list_2d.size() << " <= 5), skipping!" << std::endl;
      std::cerr << "[extract_node_semantic] Debug: dis_threshold_min=" << dis_threshold_min 
                << ", dis_threshold_max=" << dis_threshold_max << std::endl;
    }
    return;
  }
  
  if (print_debug_info_) {
    std::cerr << "[extract_node_semantic] Debug: point_list_2d size: " << point_list_2d.size() << std::endl;
  }
  
  for (auto pi : point_list_2d) {
    if (pi[0] < min_x) {
      min_x = pi[0];
    }
    if (pi[0] > max_x) {
      max_x = pi[0];
    }
    if (pi[1] < min_y) {
      min_y = pi[1];
    }
    if (pi[1] > max_y) {
      max_y = pi[1];
    }
  }
  // segment project cloud
  int segmen_base_num = 5;
  double segmen_len = segmen_base_num * resolution;
  int x_segment_num = (max_x - min_x) / segmen_len + 1;
  int y_segment_num = (max_y - min_y) / segmen_len + 1;
  int x_axis_len = (int)((max_x - min_x) / resolution + segmen_base_num);
  int y_axis_len = (int)((max_y - min_y) / resolution + segmen_base_num);

  std::vector<double> **dis_container = new std::vector<double> *[x_axis_len];
  BinaryDescriptor **binary_container = new BinaryDescriptor *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    dis_container[i] = new std::vector<double>[y_axis_len];
    binary_container[i] = new BinaryDescriptor[y_axis_len];
  }
  double **img_count = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    img_count[i] = new double[y_axis_len];
  }
  double **dis_array = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    dis_array[i] = new double[y_axis_len];
  }
  double **mean_x_list = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    mean_x_list[i] = new double[y_axis_len];
  }
  double **mean_y_list = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    mean_y_list[i] = new double[y_axis_len];
  }
  // 每个网格内点的 3D 坐标累加，用于角点取 3D 质心（保留空间高度，避免节点和三角形全落在投影平面）
  double **sum_x_3d = new double *[x_axis_len];
  double **sum_y_3d = new double *[x_axis_len];
  double **sum_z_3d = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    sum_x_3d[i] = new double[y_axis_len];
    sum_y_3d[i] = new double[y_axis_len];
    sum_z_3d[i] = new double[y_axis_len];
  }
  // 添加语义标签统计
  std::unordered_map<uint32_t, int> **label_count_map = new std::unordered_map<uint32_t, int> *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    label_count_map[i] = new std::unordered_map<uint32_t, int>[y_axis_len];
  }
  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      img_count[x][y] = 0;
      mean_x_list[x][y] = 0;
      mean_y_list[x][y] = 0;
      sum_x_3d[x][y] = 0;
      sum_y_3d[x][y] = 0;
      sum_z_3d[x][y] = 0;
      dis_array[x][y] = 0;
      std::vector<double> single_dis_container;
      dis_container[x][y] = single_dis_container;
    }
  }

  for (size_t i = 0; i < point_list_2d.size(); i++) {
    int x_index = (int)((point_list_2d[i][0] - min_x) / resolution);
    int y_index = (int)((point_list_2d[i][1] - min_y) / resolution);
    mean_x_list[x_index][y_index] += point_list_2d[i][0];
    mean_y_list[x_index][y_index] += point_list_2d[i][1];
    sum_x_3d[x_index][y_index] += point_list_3d.points[i].x;
    sum_y_3d[x_index][y_index] += point_list_3d.points[i].y;
    sum_z_3d[x_index][y_index] += point_list_3d.points[i].z;
    img_count[x_index][y_index]++;
    dis_container[x_index][y_index].push_back(dis_list_2d[i]);
    label_count_map[x_index][y_index][label_list[i]]++;  // 统计标签
  }

  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      // calc segment dis array
      if (img_count[x][y] > 0) {
        int cut_num = (dis_threshold_max - dis_threshold_min) / high_inc;
        std::vector<bool> occup_list;
        std::vector<double> cnt_list;
        BinaryDescriptor single_binary;
        for (size_t i = 0; i < cut_num; i++) {
          cnt_list.push_back(0);
          occup_list.push_back(false);
        }
        for (size_t j = 0; j < dis_container[x][y].size(); j++) {
          int cnt_index =
              (dis_container[x][y][j] - dis_threshold_min) / high_inc;
          cnt_list[cnt_index]++;
        }
        double segmnt_dis = 0;
        for (size_t i = 0; i < cut_num; i++) {
          if (cnt_list[i] >= 1) {
            segmnt_dis++;
            occup_list[i] = true;
          }
        }
        dis_array[x][y] = segmnt_dis;
        // 已废弃 binary 用于相似度匹配，不再存储 occupy_array_；summary_ 仍用于 NMS 角点筛选
        single_binary.occupy_array_.assign(1, true);
        single_binary.summary_ = static_cast<unsigned char>(segmnt_dis);
        // 找到出现频率最高和第二高的语义标签
        uint32_t most_frequent_label = 0;
        uint32_t second_frequent_label = 0;
        int max_count = 0;
        int second_max_count = 0;
        for (const auto& label_pair : label_count_map[x][y]) {
          if (label_pair.second > max_count) {
            second_max_count = max_count;
            second_frequent_label = most_frequent_label;
            max_count = label_pair.second;
            most_frequent_label = label_pair.first;
          } else if (label_pair.second > second_max_count) {
            second_max_count = label_pair.second;
            second_frequent_label = label_pair.first;
          }
        }
        single_binary.semantic_label_ = most_frequent_label;
        single_binary.semantic_ratio_ = (img_count[x][y] > 0) ? static_cast<double>(max_count) / img_count[x][y] : 0.0;
        single_binary.semantic_label_2_ = second_frequent_label;
        single_binary.semantic_ratio_2_ = (img_count[x][y] > 0) ? static_cast<double>(second_max_count) / img_count[x][y] : 0.0;
        binary_container[x][y] = single_binary;
      }
    }
  }

  // filter by distance
  std::vector<double> max_dis_list;
  std::vector<int> max_dis_x_index_list;
  std::vector<int> max_dis_y_index_list;
  
  int cut_num = (dis_threshold_max - dis_threshold_min) / high_inc;
  double max_segmt_dis_found = 0;
  int candidates_before_filter = 0;

  for (int x_segment_index = 0; x_segment_index < x_segment_num;
       x_segment_index++) {
    for (int y_segment_index = 0; y_segment_index < y_segment_num;
         y_segment_index++) {
      double max_dis = 0;
      int max_dis_x_index = -10;
      int max_dis_y_index = -10;
      for (int x_index = x_segment_index * segmen_base_num;
           x_index < (x_segment_index + 1) * segmen_base_num; x_index++) {
        for (int y_index = y_segment_index * segmen_base_num;
             y_index < (y_segment_index + 1) * segmen_base_num; y_index++) {
          if (dis_array[x_index][y_index] > max_dis) {
            max_dis = dis_array[x_index][y_index];
            max_dis_x_index = x_index;
            max_dis_y_index = y_index;
          }
        }
      }
      if (max_dis > 0) {
        candidates_before_filter++;
        max_segmt_dis_found = std::max(max_segmt_dis_found, max_dis);
      }
      if (max_dis >= binary_min_dis) {
        max_dis_list.push_back(max_dis);
        max_dis_x_index_list.push_back(max_dis_x_index);
        max_dis_y_index_list.push_back(max_dis_y_index);
      }
    }
  }
  
  if (print_debug_info_) {
    std::cerr << "[extract_node_semantic] Debug: cut_num=" << cut_num 
              << ", binary_min_dis=" << binary_min_dis 
              << ", max_segmt_dis_found=" << max_segmt_dis_found
              << ", candidates_before_filter=" << candidates_before_filter
              << ", candidates_after_filter=" << max_dis_list.size() << std::endl;
  }
  
  // calc line or not
  std::vector<Eigen::Vector2i> direction_list;
  Eigen::Vector2i d(0, 1);
  direction_list.push_back(d);
  d << 1, 0;
  direction_list.push_back(d);
  d << 1, 1;
  direction_list.push_back(d);
  d << 1, -1;
  direction_list.push_back(d);
  
  int before_line_filter = max_dis_list.size();
  int edge_filtered = 0;
  
  for (size_t i = 0; i < max_dis_list.size(); i++) {
    Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
    if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 ||
        p[1] >= y_axis_len - 1) {
      edge_filtered++;
      continue;
    }
    bool is_add = true;

    if (line_filter_enable) {
      for (int j = 0; j < 4; j++) {
        Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
        if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 ||
            p[1] >= y_axis_len - 1) {
          continue;
        }
        Eigen::Vector2i p1 = p + direction_list[j];
        Eigen::Vector2i p2 = p - direction_list[j];
        double threshold = dis_array[p[0]][p[1]] - 3;
        if (dis_array[p1[0]][p1[1]] >= threshold) {
          if (dis_array[p2[0]][p2[1]] >= 0.5 * dis_array[p[0]][p[1]]) {
            is_add = false;
          }
        }
        if (dis_array[p2[0]][p2[1]] >= threshold) {
          if (dis_array[p1[0]][p1[1]] >= 0.5 * dis_array[p[0]][p[1]]) {
            is_add = false;
          }
        }
        if (dis_array[p1[0]][p1[1]] >= threshold) {
          if (dis_array[p2[0]][p2[1]] >= threshold) {
            is_add = false;
          }
        }
        if (dis_array[p2[0]][p2[1]] >= threshold) {
          if (dis_array[p1[0]][p1[1]] >= threshold) {
            is_add = false;
          }
        }
      }
    }
    if (is_add) {
      int xi = max_dis_x_index_list[i];
      int yi = max_dis_y_index_list[i];
      double cnt = img_count[xi][yi];
      // 使用网格内点的 3D 质心作为角点位置，保留真实高度，使节点和三角形在空间中分布而非共面
      Eigen::Vector3d coord(
          sum_x_3d[xi][yi] / cnt,
          sum_y_3d[xi][yi] / cnt,
          sum_z_3d[xi][yi] / cnt);
      pcl::PointXYZ pi;
      pi.x = coord[0];
      pi.y = coord[1];
      pi.z = coord[2];
      BinaryDescriptor single_binary = binary_container[xi][yi];
      single_binary.location_ = coord;
      // 平面node：用投影平面正上方体素列的最多标签序列表示
      double cell_min_x = min_x + xi * resolution;
      double cell_max_x = min_x + (xi + 1) * resolution;
      double cell_min_y = min_y + yi * resolution;
      double cell_max_y = min_y + (yi + 1) * resolution;
      get_plane_node_label_sequence(cell_min_x, cell_max_x, cell_min_y, cell_max_y,
                                    A, B, C, D, ax, ay, bx, by, cx, cy, dx, dy,
                                    voxel_map, single_binary.label_sequence_);
      if (!single_binary.label_sequence_.empty()) {
        single_binary.semantic_label_ = single_binary.label_sequence_.front();
      }
      // 更新语义标签比例（从体素图中获取更准确的值）
      single_binary.semantic_ratio_ = get_semantic_ratio(coord, single_binary.semantic_label_, voxel_map);
      if (single_binary.semantic_label_2_ != 0) {
        single_binary.semantic_ratio_2_ = get_semantic_ratio(coord, single_binary.semantic_label_2_, voxel_map);
      }
      binary_list.push_back(single_binary);
      if (out_voxels) {
        std::vector<VOXEL_LOC> col;
        get_plane_node_voxel_column(cell_min_x, cell_max_x, cell_min_y, cell_max_y,
                                    A, B, C, D, ax, ay, bx, by, cx, cy, dx, dy,
                                    voxel_map, col);
        out_voxels->push_back(col);
      }
    }
  }
  
  if (print_debug_info_) {
    std::cerr << "[extract_node_semantic] Debug: After line filter, binary_list size: " 
              << binary_list.size() 
              << ", edge_filtered=" << edge_filtered
              << ", before_line_filter=" << before_line_filter << std::endl;
  }
  
  for (int i = 0; i < x_axis_len; i++) {
    delete[] binary_container[i];
    delete[] dis_container[i];
    delete[] img_count[i];
    delete[] dis_array[i];
    delete[] mean_x_list[i];
    delete[] mean_y_list[i];
    delete[] sum_x_3d[i];
    delete[] sum_y_3d[i];
    delete[] sum_z_3d[i];
    delete[] label_count_map[i];
  }
  delete[] binary_container;
  delete[] dis_container;
  delete[] img_count;
  delete[] dis_array;
  delete[] mean_x_list;
  delete[] mean_y_list;
  delete[] sum_x_3d;
  delete[] sum_y_3d;
  delete[] sum_z_3d;
  delete[] label_count_map;
}

void BtcDescManager::non_maxi_suppression(
    std::vector<BinaryDescriptor> &binary_list,
    std::vector<std::vector<VOXEL_LOC>> *voxels) {
  // 检查二进制描述子列表是否为空
  if (binary_list.empty()) {
    return;
  }
  if (voxels && voxels->size() != binary_list.size()) {
    voxels = nullptr;  // 长度不匹配则忽略
  }
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr prepare_key_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
  std::vector<int> pre_count_list;
  std::vector<bool> is_add_list;
  for (auto var : binary_list) {
    pcl::PointXYZ pi;
    pi.x = var.location_[0];
    pi.y = var.location_[1];
    pi.z = var.location_[2];
    prepare_key_cloud->push_back(pi);
    pre_count_list.push_back(var.summary_);
    is_add_list.push_back(true);
  }
  
  // 检查prepare_key_cloud是否为空（虽然理论上不应该，但为了安全）
  if (prepare_key_cloud->empty()) {
    if (print_debug_info_) {
      std::cerr << "[non_maxi_suppression] Warning: prepare_key_cloud is empty!" << std::endl;
    }
    return;
  }
  
  kd_tree.setInputCloud(prepare_key_cloud);
  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;
  double radius = config_setting_.non_max_suppression_radius_;
  
  if (print_debug_info_) {
    std::cerr << "[non_maxi_suppression] Debug: Using radius=" << radius 
              << " for NMS, input binary_list size=" << binary_list.size() << std::endl;
  }
  
  for (size_t i = 0; i < prepare_key_cloud->size(); i++) {
    pcl::PointXYZ searchPoint = prepare_key_cloud->points[i];
    if (kd_tree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch,
                             pointRadiusSquaredDistance) > 0) {
      Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
      for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j) {
        Eigen::Vector3d pj(
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].x,
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].y,
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].z);
        if (pointIdxRadiusSearch[j] == i) {
          continue;
        }
        // 非极大值抑制：如果当前点的summary小于邻居，则抑制当前点
        // 如果summary相等，保留索引较小的点（避免互相抑制导致都删除）
        if (pre_count_list[i] < pre_count_list[pointIdxRadiusSearch[j]] ||
            (pre_count_list[i] == pre_count_list[pointIdxRadiusSearch[j]] && 
             i > pointIdxRadiusSearch[j])) {
          is_add_list[i] = false;
        }
      }
    }
  }
  std::vector<BinaryDescriptor> pass_binary_list;
  std::vector<std::vector<VOXEL_LOC>> pass_voxels;
  int suppressed_count = 0;
  for (size_t i = 0; i < is_add_list.size(); i++) {
    if (is_add_list[i]) {
      pass_binary_list.push_back(binary_list[i]);
      if (voxels) pass_voxels.push_back((*voxels)[i]);
    } else {
      suppressed_count++;
    }
  }
  
  if (print_debug_info_) {
    std::cerr << "[non_maxi_suppression] Debug: Suppressed " << suppressed_count 
              << " descriptors, remaining " << pass_binary_list.size() 
              << " descriptors (radius=" << radius << ")" << std::endl;
  }
  
  binary_list = std::move(pass_binary_list);
  if (voxels) *voxels = std::move(pass_voxels);
}

void BtcDescManager::generate_btc(
    const std::vector<BinaryDescriptor> &binary_list, const int &frame_id,
    std::vector<BTC> &btc_list) {
  btc_list.clear();
  
  // 检查二进制描述子列表是否为空
  if (binary_list.empty()) {
    if (print_debug_info_) {
      std::cerr << "[generate_btc] Warning: binary_list is empty!" << std::endl;
    }
    return;
  }
  
  double scale = 1.0 / config_setting_.std_side_resolution_;
  std::unordered_map<VOXEL_LOC, bool> feat_map;
  pcl::PointCloud<pcl::PointXYZ> key_cloud;
  for (auto var : binary_list) {
    pcl::PointXYZ pi;
    pi.x = var.location_[0];
    pi.y = var.location_[1];
    pi.z = var.location_[2];
    key_cloud.push_back(pi);
  }
  
  // 检查key_cloud是否为空（虽然理论上不应该，但为了安全）
  if (key_cloud.empty()) {
    if (print_debug_info_) {
      std::cerr << "[generate_btc] Warning: key_cloud is empty!" << std::endl;
    }
    return;
  }
  
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  kd_tree->setInputCloud(key_cloud.makeShared());
  double search_radius = (config_setting_.descriptor_search_radius_ > 0)
                             ? config_setting_.descriptor_search_radius_
                             : config_setting_.descriptor_max_len_;
  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;
  for (size_t i = 0; i < key_cloud.size(); i++) {
    pcl::PointXYZ searchPoint = key_cloud.points[i];
    int found = kd_tree->radiusSearch(searchPoint, search_radius,
                                      pointIdxRadiusSearch, pointRadiusSquaredDistance);
    if (found > 0) {
      for (size_t m = 0; m < pointIdxRadiusSearch.size(); m++) {
        if (pointIdxRadiusSearch[m] == static_cast<int>(i)) continue;
        for (size_t n = m + 1; n < pointIdxRadiusSearch.size(); n++) {
          if (pointIdxRadiusSearch[n] == static_cast<int>(i)) continue;
          pcl::PointXYZ p1 = searchPoint;
          pcl::PointXYZ p2 = key_cloud.points[pointIdxRadiusSearch[m]];
          pcl::PointXYZ p3 = key_cloud.points[pointIdxRadiusSearch[n]];
          double a = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) +
                          pow(p1.z - p2.z, 2));
          double b = sqrt(pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2) +
                          pow(p1.z - p3.z, 2));
          double c = sqrt(pow(p3.x - p2.x, 2) + pow(p3.y - p2.y, 2) +
                          pow(p3.z - p2.z, 2));
          if (a > config_setting_.descriptor_max_len_ ||
              b > config_setting_.descriptor_max_len_ ||
              c > config_setting_.descriptor_max_len_ ||
              a < config_setting_.descriptor_min_len_ ||
              b < config_setting_.descriptor_min_len_ ||
              c < config_setting_.descriptor_min_len_) {
            continue;
          }
          double temp;
          Eigen::Vector3d A, B, C;
          Eigen::Vector3i l1, l2, l3;
          Eigen::Vector3i l_temp;
          l1 << 1, 2, 0;
          l2 << 1, 0, 3;
          l3 << 0, 2, 3;
          if (a > b) {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          if (b > c) {
            temp = b;
            b = c;
            c = temp;
            l_temp = l2;
            l2 = l3;
            l3 = l_temp;
          }
          if (a > b) {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          if (fabs(c - (a + b)) < 0.2) {
            continue;
          }

          pcl::PointXYZ d_p;
          d_p.x = a * 1000;
          d_p.y = b * 1000;
          d_p.z = c * 1000;
          VOXEL_LOC position((int64_t)d_p.x, (int64_t)d_p.y, (int64_t)d_p.z);
          auto iter = feat_map.find(position);
          Eigen::Vector3d normal_1, normal_2, normal_3;
          BinaryDescriptor binary_A;
          BinaryDescriptor binary_B;
          BinaryDescriptor binary_C;
          if (iter == feat_map.end()) {
            if (l1[0] == l2[0]) {
              A << p1.x, p1.y, p1.z;
              binary_A = binary_list[i];
            } else if (l1[1] == l2[1]) {
              A << p2.x, p2.y, p2.z;
              binary_A = binary_list[pointIdxRadiusSearch[m]];
            } else {
              A << p3.x, p3.y, p3.z;
              binary_A = binary_list[pointIdxRadiusSearch[n]];
            }
            if (l1[0] == l3[0]) {
              B << p1.x, p1.y, p1.z;
              binary_B = binary_list[i];
            } else if (l1[1] == l3[1]) {
              B << p2.x, p2.y, p2.z;
              binary_B = binary_list[pointIdxRadiusSearch[m]];
            } else {
              B << p3.x, p3.y, p3.z;
              binary_B = binary_list[pointIdxRadiusSearch[n]];
            }
            if (l2[0] == l3[0]) {
              C << p1.x, p1.y, p1.z;
              binary_C = binary_list[i];
            } else if (l2[1] == l3[1]) {
              C << p2.x, p2.y, p2.z;
              binary_C = binary_list[pointIdxRadiusSearch[m]];
            } else {
              C << p3.x, p3.y, p3.z;
              binary_C = binary_list[pointIdxRadiusSearch[n]];
            }
            BTC single_descriptor;
            single_descriptor.binary_A_ = binary_A;
            single_descriptor.binary_B_ = binary_B;
            single_descriptor.binary_C_ = binary_C;
            single_descriptor.center_ = (A + B + C) / 3;
            single_descriptor.triangle_ << scale * a, scale * b, scale * c;
            single_descriptor.angle_[0] = fabs(5 * normal_1.dot(normal_2));
            single_descriptor.angle_[1] = fabs(5 * normal_1.dot(normal_3));
            single_descriptor.angle_[2] = fabs(5 * normal_3.dot(normal_2));
            // single_descriptor.angle << 0, 0, 0;
            single_descriptor.frame_number_ = frame_id;
            // 向后兼容：如果没有语义标签，设置为0
            single_descriptor.vertex_semantic_ << 0.0, 0.0, 0.0;
            single_descriptor.vertex_semantic_ratio_ << 0.0, 0.0, 0.0;
            // single_descriptor.score_frame_.push_back(frame_number);
            Eigen::Matrix3d triangle_positon;
            triangle_positon.block<3, 1>(0, 0) = A;
            triangle_positon.block<3, 1>(0, 1) = B;
            triangle_positon.block<3, 1>(0, 2) = C;
            // single_descriptor.position_list_.push_back(triangle_positon);
            // single_descriptor.triangle_scale_ = scale;
            feat_map[position] = true;
            btc_list.push_back(single_descriptor);
          }
        }
      }
    }
  }
}

void SemanticTriangularDescManager::generate_semantic_triangular_desc(
    const std::vector<BinaryDescriptor> &binary_list, const int &frame_id,
    std::vector<SemanticTriangularDescriptor> &std_list) {
  std_list.clear();
  
  // 检查二进制描述子列表是否为空
  if (binary_list.empty()) {
    if (print_debug_info_) {
      std::cerr << "[generate_semantic_triangular_desc] Warning: binary_list is empty!" << std::endl;
    }
    return;
  }
  
  double scale = 1.0 / config_setting_.std_side_resolution_;
  std::unordered_map<VOXEL_LOC, bool> feat_map;
  pcl::PointCloud<pcl::PointXYZ> key_cloud;
  for (auto var : binary_list) {
    pcl::PointXYZ pi;
    pi.x = var.location_[0];
    pi.y = var.location_[1];
    pi.z = var.location_[2];
    key_cloud.push_back(pi);
  }
  
  // 检查key_cloud是否为空（虽然理论上不应该，但为了安全）
  if (key_cloud.empty()) {
    if (print_debug_info_) {
      std::cerr << "[generate_semantic_triangular_desc] Warning: key_cloud is empty!" << std::endl;
    }
    return;
  }
  
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  kd_tree->setInputCloud(key_cloud.makeShared());
  double search_radius = (config_setting_.descriptor_search_radius_ > 0)
                             ? config_setting_.descriptor_search_radius_
                             : config_setting_.descriptor_max_len_;
  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;
  for (size_t i = 0; i < key_cloud.size(); i++) {
    pcl::PointXYZ searchPoint = key_cloud.points[i];
    int found = kd_tree->radiusSearch(searchPoint, search_radius,
                                      pointIdxRadiusSearch, pointRadiusSquaredDistance);
    if (found > 0) {
      for (size_t m = 0; m < pointIdxRadiusSearch.size(); m++) {
        if (pointIdxRadiusSearch[m] == static_cast<int>(i)) continue;
        for (size_t n = m + 1; n < pointIdxRadiusSearch.size(); n++) {
          if (pointIdxRadiusSearch[n] == static_cast<int>(i)) continue;
          pcl::PointXYZ p1 = searchPoint;
          pcl::PointXYZ p2 = key_cloud.points[pointIdxRadiusSearch[m]];
          pcl::PointXYZ p3 = key_cloud.points[pointIdxRadiusSearch[n]];
          double a = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) +
                          pow(p1.z - p2.z, 2));
          double b = sqrt(pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2) +
                          pow(p1.z - p3.z, 2));
          double c = sqrt(pow(p3.x - p2.x, 2) + pow(p3.y - p2.y, 2) +
                          pow(p3.z - p2.z, 2));
          if (a > config_setting_.descriptor_max_len_ ||
              b > config_setting_.descriptor_max_len_ ||
              c > config_setting_.descriptor_max_len_ ||
              a < config_setting_.descriptor_min_len_ ||
              b < config_setting_.descriptor_min_len_ ||
              c < config_setting_.descriptor_min_len_) {
            continue;
          }
          double temp;
          Eigen::Vector3d A, B, C;
          Eigen::Vector3i l1, l2, l3;
          Eigen::Vector3i l_temp;
          l1 << 1, 2, 0;
          l2 << 1, 0, 3;
          l3 << 0, 2, 3;
          if (a > b) {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          if (b > c) {
            temp = b;
            b = c;
            c = temp;
            l_temp = l2;
            l2 = l3;
            l3 = l_temp;
          }
          if (a > b) {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          if (fabs(c - (a + b)) < 0.2) {
            continue;
          }

          pcl::PointXYZ d_p;
          d_p.x = a * 1000;
          d_p.y = b * 1000;
          d_p.z = c * 1000;
          VOXEL_LOC position((int64_t)d_p.x, (int64_t)d_p.y, (int64_t)d_p.z);
          auto iter = feat_map.find(position);
          Eigen::Vector3d normal_1, normal_2, normal_3;
          BinaryDescriptor binary_A;
          BinaryDescriptor binary_B;
          BinaryDescriptor binary_C;
          if (iter == feat_map.end()) {
            if (l1[0] == l2[0]) {
              A << p1.x, p1.y, p1.z;
              binary_A = binary_list[i];
            } else if (l1[1] == l2[1]) {
              A << p2.x, p2.y, p2.z;
              binary_A = binary_list[pointIdxRadiusSearch[m]];
            } else {
              A << p3.x, p3.y, p3.z;
              binary_A = binary_list[pointIdxRadiusSearch[n]];
            }
            if (l1[0] == l3[0]) {
              B << p1.x, p1.y, p1.z;
              binary_B = binary_list[i];
            } else if (l1[1] == l3[1]) {
              B << p2.x, p2.y, p2.z;
              binary_B = binary_list[pointIdxRadiusSearch[m]];
            } else {
              B << p3.x, p3.y, p3.z;
              binary_B = binary_list[pointIdxRadiusSearch[n]];
            }
            if (l2[0] == l3[0]) {
              C << p1.x, p1.y, p1.z;
              binary_C = binary_list[i];
            } else if (l2[1] == l3[1]) {
              C << p2.x, p2.y, p2.z;
              binary_C = binary_list[pointIdxRadiusSearch[m]];
            } else {
              C << p3.x, p3.y, p3.z;
              binary_C = binary_list[pointIdxRadiusSearch[n]];
            }
            SemanticTriangularDescriptor single_descriptor;
            single_descriptor.binary_A_ = binary_A;
            single_descriptor.binary_B_ = binary_B;
            single_descriptor.binary_C_ = binary_C;
            single_descriptor.center_ = (A + B + C) / 3;
            single_descriptor.triangle_ << scale * a, scale * b, scale * c;
            single_descriptor.angle_[0] = fabs(5 * normal_1.dot(normal_2));
            single_descriptor.angle_[1] = fabs(5 * normal_1.dot(normal_3));
            single_descriptor.angle_[2] = fabs(5 * normal_3.dot(normal_2));
            single_descriptor.frame_number_ = frame_id;
            // 存储三个顶点的语义标签（按边长顺序：A, B, C）- 第一标签
            // 注意：A, B, C 的语义标签必须与 binary_A, binary_B, binary_C 一一对应
            // 这确保了在 candidate_selector 中语义匹配时，vertex_semantic_[0/1/2] 对应 A/B/C
            // 在 candidate_verify 和 triangle_solver 中，binary_A/B/C.location_ 也对应 A/B/C
            // 如果没有设置语义标签，默认为0（向后兼容）
            single_descriptor.vertex_semantic_ << static_cast<double>(binary_A.semantic_label_),
                                                 static_cast<double>(binary_B.semantic_label_),
                                                 static_cast<double>(binary_C.semantic_label_);
            // 存储三个顶点的语义标签比重（按边长顺序：A, B, C，范围0-1）- 第一标签
            // 如果没有设置语义比例，默认为0（向后兼容）
            single_descriptor.vertex_semantic_ratio_ << binary_A.semantic_ratio_,
                                                        binary_B.semantic_ratio_,
                                                        binary_C.semantic_ratio_;
            // 存储三个顶点的语义标签（按边长顺序：A, B, C）- 第二标签
            single_descriptor.vertex_semantic_2_ << static_cast<double>(binary_A.semantic_label_2_),
                                                    static_cast<double>(binary_B.semantic_label_2_),
                                                    static_cast<double>(binary_C.semantic_label_2_);
            // 存储三个顶点的语义标签比重（按边长顺序：A, B, C，范围0-1）- 第二标签
            single_descriptor.vertex_semantic_ratio_2_ << binary_A.semantic_ratio_2_,
                                                           binary_B.semantic_ratio_2_,
                                                           binary_C.semantic_ratio_2_;
            Eigen::Matrix3d triangle_positon;
            triangle_positon.block<3, 1>(0, 0) = A;
            triangle_positon.block<3, 1>(0, 1) = B;
            triangle_positon.block<3, 1>(0, 2) = C;
            feat_map[position] = true;
            std_list.push_back(single_descriptor);
          }
        }
      }
    }
  }
}

// 平面node标签序列相似度：按位置对齐，2*匹配数/(L1+L2)，超过阈值视为匹配
static double sequence_similarity(const std::vector<uint32_t> &a,
                                  const std::vector<uint32_t> &b) {
  if (a.empty() && b.empty()) return 1.0;
  if (a.empty() || b.empty()) return 0.0;
  size_t n = std::min(a.size(), b.size());
  size_t match = 0;
  for (size_t i = 0; i < n; i++) {
    if (a[i] == b[i]) match++;
  }
  return 2.0 * static_cast<double>(match) / static_cast<double>(a.size() + b.size());
}

// Note: BtcDescManager::candidate_selector with BTC type is the same as SemanticTriangularDescManager::candidate_selector
// since BTC is a typedef of SemanticTriangularDescriptor. However, the backward compatibility version
// doesn't check semantic labels. We use the semantic version which can handle both cases (with or without semantic labels).

void SemanticTriangularDescManager::candidate_selector(
    const std::vector<SemanticTriangularDescriptor> &current_STD_list,
    std::vector<SemanticTriangularMatchList> &candidate_matcher_vec) {
  int current_frame_id = current_STD_list[0].frame_number_;
  double match_array[20000] = {0};
  std::vector<int> match_list_index;
  std::vector<Eigen::Vector3i> voxel_round;
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        Eigen::Vector3i voxel_inc(x, y, z);
        voxel_round.push_back(voxel_inc);
      }
    }
  }
  std::vector<bool> useful_match(current_STD_list.size());
  std::vector<std::vector<size_t>> useful_match_index(current_STD_list.size());
  std::vector<std::vector<SemanticTriangularDescriptor_LOC>> useful_match_position(
      current_STD_list.size());
  std::vector<size_t> index(current_STD_list.size());
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
    useful_match[i] = false;
  }
  std::mutex mylock;
  auto t0 = std::chrono::high_resolution_clock::now();

  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i) {
        SemanticTriangularDescriptor descriptor = current_STD_list[i];
        SemanticTriangularDescriptor_LOC position;
        double dis_threshold =
            descriptor.triangle_.norm() *
            config_setting_.rough_dis_threshold_;
        for (auto voxel_inc : voxel_round) {
          position.x = (int)(descriptor.triangle_[0] + voxel_inc[0]);
          position.y = (int)(descriptor.triangle_[1] + voxel_inc[1]);
          position.z = (int)(descriptor.triangle_[2] + voxel_inc[2]);
          Eigen::Vector3d voxel_center((double)position.x + 0.5,
                                       (double)position.y + 0.5,
                                       (double)position.z + 0.5);
          if ((descriptor.triangle_ - voxel_center).norm() < 1.5) {
            auto iter = data_base_.find(position);
            if (iter != data_base_.end()) {
              for (size_t j = 0; j < data_base_[position].size(); j++) {
                // 去除帧序列号间隔限制（用于测试匹配性能，允许匹配相邻帧）
                // 原代码：if ((descriptor.frame_number_ - data_base_[position][j].frame_number_) > config_setting_.skip_near_num_)
                // 现在：允许所有帧匹配（包括相邻帧）
                  double dis =
                      (descriptor.triangle_ - data_base_[position][j].triangle_)
                          .norm();
                  if (dis < dis_threshold) {
                  // 检查语义标签匹配（按边长顺序：A, B, C）
                  // 允许src的第一/第二标签与tgt的第一/第二标签分别匹配
                  uint32_t src_sem_A_1 = static_cast<uint32_t>(descriptor.vertex_semantic_[0] + 0.5);
                  uint32_t src_sem_A_2 = static_cast<uint32_t>(descriptor.vertex_semantic_2_[0] + 0.5);
                  uint32_t src_sem_B_1 = static_cast<uint32_t>(descriptor.vertex_semantic_[1] + 0.5);
                  uint32_t src_sem_B_2 = static_cast<uint32_t>(descriptor.vertex_semantic_2_[1] + 0.5);
                  uint32_t src_sem_C_1 = static_cast<uint32_t>(descriptor.vertex_semantic_[2] + 0.5);
                  uint32_t src_sem_C_2 = static_cast<uint32_t>(descriptor.vertex_semantic_2_[2] + 0.5);
                  
                  uint32_t tgt_sem_A_1 = static_cast<uint32_t>(data_base_[position][j].vertex_semantic_[0] + 0.5);
                  uint32_t tgt_sem_A_2 = static_cast<uint32_t>(data_base_[position][j].vertex_semantic_2_[0] + 0.5);
                  uint32_t tgt_sem_B_1 = static_cast<uint32_t>(data_base_[position][j].vertex_semantic_[1] + 0.5);
                  uint32_t tgt_sem_B_2 = static_cast<uint32_t>(data_base_[position][j].vertex_semantic_2_[1] + 0.5);
                  uint32_t tgt_sem_C_1 = static_cast<uint32_t>(data_base_[position][j].vertex_semantic_[2] + 0.5);
                  uint32_t tgt_sem_C_2 = static_cast<uint32_t>(data_base_[position][j].vertex_semantic_2_[2] + 0.5);
                  
                  // 检查是否有语义标签（如果所有标签都是0，则认为是向后兼容模式，跳过语义检查）
                  bool has_semantic_info = (src_sem_A_1 != 0 || src_sem_A_2 != 0 || src_sem_B_1 != 0 || src_sem_B_2 != 0 || src_sem_C_1 != 0 || src_sem_C_2 != 0) ||
                                           (tgt_sem_A_1 != 0 || tgt_sem_A_2 != 0 || tgt_sem_B_1 != 0 || tgt_sem_B_2 != 0 || tgt_sem_C_1 != 0 || tgt_sem_C_2 != 0);
                  
                  // 如果设置了语义匹配阈值（>0），则进行语义检查
                  // 如果阈值为0，则跳过语义检查（向后兼容模式）
                  if (has_semantic_info && config_setting_.semantic_vertex_match_threshold_ > 0) {
                    // 辅助函数：检查单个顶点是否匹配
                    // 对于每个顶点，检查src的第一/第二标签是否与tgt的第一/第二标签匹配
                    // 只要有一组标签匹配且比例差异<阈值，就认为该顶点匹配
                    auto check_vertex_match = [&](uint32_t src_1, uint32_t src_2, double src_ratio_1, double src_ratio_2,
                                                   uint32_t tgt_1, uint32_t tgt_2, double tgt_ratio_1, double tgt_ratio_2) -> bool {
                      // 检查4种可能的匹配组合
                      // 1. src第一标签 == tgt第一标签
                      if (src_1 != 0 && tgt_1 != 0 && src_1 == tgt_1) {
                        double ratio_diff = std::abs(src_ratio_1 - tgt_ratio_1);
                        if (ratio_diff < config_setting_.semantic_ratio_threshold_) {
                          return true;
                        }
                      }
                      // 2. src第一标签 == tgt第二标签
                      if (src_1 != 0 && tgt_2 != 0 && src_1 == tgt_2) {
                        double ratio_diff = std::abs(src_ratio_1 - tgt_ratio_2);
                        if (ratio_diff < config_setting_.semantic_ratio_threshold_) {
                          return true;
                        }
                      }
                      // 3. src第二标签 == tgt第一标签
                      if (src_2 != 0 && tgt_1 != 0 && src_2 == tgt_1) {
                        double ratio_diff = std::abs(src_ratio_2 - tgt_ratio_1);
                        if (ratio_diff < config_setting_.semantic_ratio_threshold_) {
                          return true;
                        }
                      }
                      // 4. src第二标签 == tgt第二标签
                      if (src_2 != 0 && tgt_2 != 0 && src_2 == tgt_2) {
                        double ratio_diff = std::abs(src_ratio_2 - tgt_ratio_2);
                        if (ratio_diff < config_setting_.semantic_ratio_threshold_) {
                          return true;
                        }
                      }
                      return false;
                    };
                    
                    // 检查三个顶点是否匹配：平面node有标签序列则用序列比对，否则用单标签+比例
                    auto vertex_match = [&](const BinaryDescriptor &src_bin, const BinaryDescriptor &tgt_bin,
                                            uint32_t s1, uint32_t s2, double sr1, double sr2,
                                            uint32_t t1, uint32_t t2, double tr1, double tr2) -> bool {
                      if (!src_bin.label_sequence_.empty() && !tgt_bin.label_sequence_.empty()) {
                        return sequence_similarity(src_bin.label_sequence_, tgt_bin.label_sequence_) >=
                               config_setting_.plane_node_sequence_similarity_threshold_;
                      }
                      return check_vertex_match(s1, s2, sr1, sr2, t1, t2, tr1, tr2);
                    };
                    bool vertex_A_matches = vertex_match(
                        descriptor.binary_A_, data_base_[position][j].binary_A_,
                        src_sem_A_1, src_sem_A_2,
                        descriptor.vertex_semantic_ratio_[0], descriptor.vertex_semantic_ratio_2_[0],
                        tgt_sem_A_1, tgt_sem_A_2,
                        data_base_[position][j].vertex_semantic_ratio_[0], data_base_[position][j].vertex_semantic_ratio_2_[0]);
                    bool vertex_B_matches = vertex_match(
                        descriptor.binary_B_, data_base_[position][j].binary_B_,
                        src_sem_B_1, src_sem_B_2,
                        descriptor.vertex_semantic_ratio_[1], descriptor.vertex_semantic_ratio_2_[1],
                        tgt_sem_B_1, tgt_sem_B_2,
                        data_base_[position][j].vertex_semantic_ratio_[1], data_base_[position][j].vertex_semantic_ratio_2_[1]);
                    bool vertex_C_matches = vertex_match(
                        descriptor.binary_C_, data_base_[position][j].binary_C_,
                        src_sem_C_1, src_sem_C_2,
                        descriptor.vertex_semantic_ratio_[2], descriptor.vertex_semantic_ratio_2_[2],
                        tgt_sem_C_1, tgt_sem_C_2,
                        data_base_[position][j].vertex_semantic_ratio_[2], data_base_[position][j].vertex_semantic_ratio_2_[2]);
                    
                    // 计算匹配的顶点数
                    int semantic_matches = 0;
                    if (vertex_A_matches) semantic_matches++;
                    if (vertex_B_matches) semantic_matches++;
                    if (vertex_C_matches) semantic_matches++;
                    
                    // 要求至少 semantic_vertex_match_threshold_ 个顶点匹配
                    if (semantic_matches < config_setting_.semantic_vertex_match_threshold_) {
                      if (print_debug_info_ && i < 5) {  // 只打印前几个，避免输出过多
                        std::cerr << "[candidate_selector] Debug: Semantic mismatch. "
                                  << "src_sem_A=(" << src_sem_A_1 << "," << src_sem_A_2 << "), "
                                  << "src_sem_B=(" << src_sem_B_1 << "," << src_sem_B_2 << "), "
                                  << "src_sem_C=(" << src_sem_C_1 << "," << src_sem_C_2 << "), "
                                  << "tgt_sem_A=(" << tgt_sem_A_1 << "," << tgt_sem_A_2 << "), "
                                  << "tgt_sem_B=(" << tgt_sem_B_1 << "," << tgt_sem_B_2 << "), "
                                  << "tgt_sem_C=(" << tgt_sem_C_1 << "," << tgt_sem_C_2 << "), "
                                  << "matches=" << semantic_matches 
                                  << ", threshold=" << config_setting_.semantic_vertex_match_threshold_ << std::endl;
                      }
                      continue;  // 语义标签匹配不足，跳过
                    }
                  }
                  // 已废弃 binary 相似度过滤（similarity_threshold=0 最优，说明 binary 相似度起反作用），仅保留几何与语义匹配
                  useful_match[i] = true;
                  useful_match_position[i].push_back(position);
                  useful_match_index[i].push_back(j);
                }
              }
            }
          }
        }
      });
  std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>>
      index_recorder;
  auto t1 = std::chrono::high_resolution_clock::now();
  int total_useful_matches = 0;
  for (size_t i = 0; i < useful_match.size(); i++) {
    if (useful_match[i]) {
      total_useful_matches += useful_match_index[i].size();
      for (size_t j = 0; j < useful_match_index[i].size(); j++) {
        match_array[data_base_[useful_match_position[i][j]]
                              [useful_match_index[i][j]]
                                  .frame_number_] += 1;
        Eigen::Vector2i match_index(i, j);
        index_recorder.push_back(match_index);
        match_list_index.push_back(
            data_base_[useful_match_position[i][j]][useful_match_index[i][j]]
                .frame_number_);
      }
    }
  }
  
  if (print_debug_info_) {
    std::cout << "[candidate_selector] Debug: total_useful_matches=" << total_useful_matches 
              << ", index_recorder.size()=" << index_recorder.size() << std::endl;
    // 输出每个帧的投票数（用于诊断）
    std::map<int, int> vote_count_map;
    for (int i = 0; i < 20000; i++) {
      if (match_array[i] > 0) {
        vote_count_map[i] = match_array[i];
      }
    }
    if (!vote_count_map.empty()) {
      std::cout << "[candidate_selector] Debug: Vote counts for frames: ";
      for (const auto& pair : vote_count_map) {
        std::cout << "frame_" << pair.first << "=" << pair.second << " ";
      }
      std::cout << std::endl;
    }
  }

  for (int cnt = 0; cnt < config_setting_.candidate_num_; cnt++) {
    double max_vote = 1;
    int max_vote_index = -1;
    for (int i = 0; i < 20000; i++) {
      if (match_array[i] > max_vote) {
        max_vote = match_array[i];
        max_vote_index = i;
      }
    }
    SemanticTriangularMatchList match_triangle_list;
    if (max_vote_index >= 0 && max_vote >= 5) {
      if (print_debug_info_) {
        std::cout << "[candidate_selector] Debug: Found candidate frame " << max_vote_index 
                  << " with " << max_vote << " votes" << std::endl;
      }
      match_array[max_vote_index] = 0;
      match_triangle_list.match_frame_ = max_vote_index;
      match_triangle_list.match_id_.first = current_frame_id;
      match_triangle_list.match_id_.second = max_vote_index;
      for (size_t i = 0; i < index_recorder.size(); i++) {
        if (match_list_index[i] == max_vote_index) {
          std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor> single_match_pair;
          single_match_pair.first = current_STD_list[index_recorder[i][0]];
          single_match_pair.second =
              data_base_[useful_match_position[index_recorder[i][0]]
                                              [index_recorder[i][1]]]
                        [useful_match_index[index_recorder[i][0]]
                                           [index_recorder[i][1]]];
          match_triangle_list.match_list_.push_back(single_match_pair);
        }
      }
      candidate_matcher_vec.push_back(match_triangle_list);
    } else if (max_vote_index >= 0) {
      // 即使不打印调试信息，也记录投票不足的候选（用于诊断）
      if (print_debug_info_) {
        std::cerr << "[candidate_selector] Debug: Candidate frame " << max_vote_index 
                  << " has only " << max_vote << " votes (need >= 5)" << std::endl;
      }
      // 记录所有投票数，即使不足阈值（用于分析）
      if (max_vote > 0 && max_vote < 5) {
        // 可以在这里添加统计信息，记录哪些帧因为投票不足而失败
      }
    }
  }
  
  if (print_debug_info_) {
    std::cout << "[candidate_selector] Debug: "
              << "current_STD_list.size()=" << current_STD_list.size()
              << ", candidate_matcher_vec.size()=" << candidate_matcher_vec.size() << std::endl;
    if (candidate_matcher_vec.empty()) {
      std::cerr << "[candidate_selector] Warning: No candidates found! "
                << "This may be due to strict semantic matching conditions." << std::endl;
    }
  }
}

// Note: BtcDescManager::candidate_verify with BTCMatchList type is the same as SemanticTriangularDescManager::candidate_verify
// since BTCMatchList is a typedef of SemanticTriangularMatchList. The semantic version implementation above handles both cases.

void SemanticTriangularDescManager::candidate_verify(
    const SemanticTriangularMatchList &candidate_matcher, double &verify_score,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &relative_pose,
    std::vector<std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor>> &sucess_match_list,
    double* pose_verify_ms, double* refine_ms) {
  sucess_match_list.clear();
  double prelim_pose_acc_ms = 0.0;
  double plane_verify_acc_ms = 0.0;
  double refine_acc_ms = 0.0;
  double dis_threshold = 3;
  int skip_len = (int)(candidate_matcher.match_list_.size() / 50) + 1;
  int use_size = candidate_matcher.match_list_.size() / skip_len;
  std::vector<size_t> index(use_size);
  std::vector<int> vote_list(use_size);
  std::vector<double> weighted_vote_list(use_size, 0.0);
  const float plane_w = config_setting_.plane_vertex_weight_;
  const float instance_w = config_setting_.instance_vertex_weight_;
  for (size_t i = 0; i < index.size(); i++) {
    index[i] = i;
  }
  std::mutex mylock;
  auto prelim_t0 = std::chrono::high_resolution_clock::now();
  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i) {
        auto single_pair = candidate_matcher.match_list_[i * skip_len];
        int vote = 0;
        double weighted_vote = 0.0;
        Eigen::Matrix3d test_rot;
        Eigen::Vector3d test_t;
        triangle_solver(single_pair, test_t, test_rot);
        for (size_t j = 0; j < candidate_matcher.match_list_.size(); j++) {
          auto verify_pair = candidate_matcher.match_list_[j];
          Eigen::Vector3d A = verify_pair.first.binary_A_.location_;
          Eigen::Vector3d A_transform = test_rot * A + test_t;
          Eigen::Vector3d B = verify_pair.first.binary_B_.location_;
          Eigen::Vector3d B_transform = test_rot * B + test_t;
          Eigen::Vector3d C = verify_pair.first.binary_C_.location_;
          Eigen::Vector3d C_transform = test_rot * C + test_t;
          double dis_A =
              (A_transform - verify_pair.second.binary_A_.location_).norm();
          double dis_B =
              (B_transform - verify_pair.second.binary_B_.location_).norm();
          double dis_C =
              (C_transform - verify_pair.second.binary_C_.location_).norm();
          if (dis_A < dis_threshold && dis_B < dis_threshold &&
              dis_C < dis_threshold) {
            vote++;
            int n_plane = static_cast<int>(!verify_pair.first.binary_A_.is_instance_node_) +
                         static_cast<int>(!verify_pair.first.binary_B_.is_instance_node_) +
                         static_cast<int>(!verify_pair.first.binary_C_.is_instance_node_);
            double w = (plane_w * n_plane + instance_w * (3 - n_plane)) / 3.0;
            weighted_vote += w;
          }
        }
        mylock.lock();
        vote_list[i] = vote;
        weighted_vote_list[i] = weighted_vote;
        mylock.unlock();
      });

  int max_vote_index = 0;
  double max_weighted_vote = 0.0;
  for (size_t i = 0; i < vote_list.size(); i++) {
    if (weighted_vote_list[i] > max_weighted_vote) {
      max_vote_index = i;
      max_weighted_vote = weighted_vote_list[i];
    }
  }
  int max_vote = (vote_list.empty() ? 0 : vote_list[max_vote_index]);

  if (max_vote >= 4) {
    auto best_pair = candidate_matcher.match_list_[max_vote_index * skip_len];
    Eigen::Matrix3d best_rot;
    Eigen::Vector3d best_t;
    triangle_solver(best_pair, best_t, best_rot);
    relative_pose.first = best_t;
    relative_pose.second = best_rot;
    for (size_t j = 0; j < candidate_matcher.match_list_.size(); j++) {
      auto verify_pair = candidate_matcher.match_list_[j];
      Eigen::Vector3d A = verify_pair.first.binary_A_.location_;
      Eigen::Vector3d A_transform = best_rot * A + best_t;
      Eigen::Vector3d B = verify_pair.first.binary_B_.location_;
      Eigen::Vector3d B_transform = best_rot * B + best_t;
      Eigen::Vector3d C = verify_pair.first.binary_C_.location_;
      Eigen::Vector3d C_transform = best_rot * C + best_t;
      double dis_A =
          (A_transform - verify_pair.second.binary_A_.location_).norm();
      double dis_B =
          (B_transform - verify_pair.second.binary_B_.location_).norm();
      double dis_C =
          (C_transform - verify_pair.second.binary_C_.location_).norm();
      if (dis_A < dis_threshold && dis_B < dis_threshold &&
          dis_C < dis_threshold) {
        sucess_match_list.push_back(verify_pair);
      }
    }
    auto prelim_t1 = std::chrono::high_resolution_clock::now();
    prelim_pose_acc_ms += std::chrono::duration<double, std::milli>(prelim_t1 - prelim_t0).count();

    if (plane_cloud_vec_.empty() || 
        candidate_matcher.match_id_.second < 0 || 
        static_cast<size_t>(candidate_matcher.match_id_.second) >= plane_cloud_vec_.size() ||
        plane_cloud_vec_.size() < 2 ||
        plane_cloud_vec_.back()->empty() ||
        plane_cloud_vec_[candidate_matcher.match_id_.second]->empty()) {
      if (print_debug_info_) {
        std::cerr << "[candidate_verify] Warning: plane_cloud is empty or invalid index! "
                  << "plane_cloud_vec_.size()=" << plane_cloud_vec_.size()
                  << ", match_id_.second=" << candidate_matcher.match_id_.second << std::endl;
      }
      verify_score = -1;
    } else {
      auto verify_t0 = std::chrono::high_resolution_clock::now();
      verify_score = plane_geometric_verify(
          plane_cloud_vec_.back(),
          plane_cloud_vec_[candidate_matcher.match_id_.second],
          relative_pose);
      auto verify_t1 = std::chrono::high_resolution_clock::now();
      plane_verify_acc_ms += std::chrono::duration<double, std::milli>(verify_t1 - verify_t0).count();

      if (verify_score > 0 && !plane_cloud_vec_.back()->empty() && 
          !plane_cloud_vec_[candidate_matcher.match_id_.second]->empty()) {
        const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> *source_semantic_map = nullptr;
        const std::unordered_map<VOXEL_LOC, std::vector<std::pair<Eigen::Vector3d, uint32_t>>> *target_semantic_map = nullptr;
        
        if (semantic_voxel_map_vec_.size() > 0) {
          size_t source_idx = semantic_voxel_map_vec_.size() - 1;
          if (source_idx < semantic_voxel_map_vec_.size()) {
            source_semantic_map = &semantic_voxel_map_vec_[source_idx];
          }
          size_t target_idx = candidate_matcher.match_id_.second;
          if (target_idx < semantic_voxel_map_vec_.size()) {
            target_semantic_map = &semantic_voxel_map_vec_[target_idx];
          }
        }
        
        auto refine_t0 = std::chrono::high_resolution_clock::now();
        PlaneGeomrtricIcp(
            plane_cloud_vec_.back(),
            plane_cloud_vec_[candidate_matcher.match_id_.second],
            relative_pose,
            source_semantic_map,
            target_semantic_map);
        auto refine_t1 = std::chrono::high_resolution_clock::now();
        refine_acc_ms += std::chrono::duration<double, std::milli>(refine_t1 - refine_t0).count();

        auto verify_t2 = std::chrono::high_resolution_clock::now();
        verify_score = plane_geometric_verify(
            plane_cloud_vec_.back(),
            plane_cloud_vec_[candidate_matcher.match_id_.second],
            relative_pose);
        auto verify_t3 = std::chrono::high_resolution_clock::now();
        plane_verify_acc_ms += std::chrono::duration<double, std::milli>(verify_t3 - verify_t2).count();
      }
    }
  } else {
    auto prelim_t1 = std::chrono::high_resolution_clock::now();
    prelim_pose_acc_ms += std::chrono::duration<double, std::milli>(prelim_t1 - prelim_t0).count();
    if (print_debug_info_) {
      std::cerr << "[candidate_verify] Debug: max_vote=" << max_vote 
                << " < 2, verify_score set to -1" << std::endl;
    }
    verify_score = -1;
  }
  if (pose_verify_ms) {
    *pose_verify_ms = prelim_pose_acc_ms + plane_verify_acc_ms;
  }
  if (refine_ms) *refine_ms = refine_acc_ms;
  return;
}

void SemanticTriangularDescManager::triangle_solver(std::pair<SemanticTriangularDescriptor, SemanticTriangularDescriptor> &std_pair,
                                     Eigen::Vector3d &t, Eigen::Matrix3d &rot) {
  Eigen::Matrix3d src = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d ref = Eigen::Matrix3d::Zero();
  src.col(0) = std_pair.first.binary_A_.location_ - std_pair.first.center_;
  src.col(1) = std_pair.first.binary_B_.location_ - std_pair.first.center_;
  src.col(2) = std_pair.first.binary_C_.location_ - std_pair.first.center_;
  ref.col(0) = std_pair.second.binary_A_.location_ - std_pair.second.center_;
  ref.col(1) = std_pair.second.binary_B_.location_ - std_pair.second.center_;
  ref.col(2) = std_pair.second.binary_C_.location_ - std_pair.second.center_;
  Eigen::Matrix3d covariance = src * ref.transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::Matrix3d V = svd.matrixV();
  Eigen::Matrix3d U = svd.matrixU();
  rot = V * U.transpose();
  if (rot.determinant() < 0) {
    Eigen::Matrix3d K;
    K << 1, 0, 0, 0, 1, 0, 0, 0, -1;
    rot = V * K * U.transpose();
  }
  t = -rot * std_pair.first.center_ + std_pair.second.center_;
}

// Note: BtcDescManager::triangle_solver with BTC type is the same as SemanticTriangularDescManager::triangle_solver
// since BTC is a typedef of SemanticTriangularDescriptor. The semantic version implementation above handles both cases.

double BtcDescManager::plane_geometric_verify(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
    const std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform) {
  // 检查点云是否为空
  if (source_cloud->empty() || target_cloud->empty()) {
    if (print_debug_info_) {
      std::cerr << "[plane_geometric_verify] Warning: source_cloud or target_cloud is empty!" << std::endl;
    }
    return 0.0;
  }
  
  Eigen::Vector3d t = transform.first;
  Eigen::Matrix3d rot = transform.second;
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < target_cloud->size(); i++) {
    pcl::PointXYZ pi;
    pi.x = target_cloud->points[i].x;
    pi.y = target_cloud->points[i].y;
    pi.z = target_cloud->points[i].z;
    input_cloud->push_back(pi);
  }

  kd_tree->setInputCloud(input_cloud);
  // 创建两个向量，分别存放近邻的索引值、近邻的中心距
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  double useful_match = 0;
  double normal_threshold = config_setting_.normal_threshold_;
  double dis_threshold = config_setting_.dis_threshold_;
  for (size_t i = 0; i < source_cloud->size(); i++) {
    pcl::PointXYZINormal searchPoint = source_cloud->points[i];
    pcl::PointXYZ use_search_point;
    use_search_point.x = searchPoint.x;
    use_search_point.y = searchPoint.y;
    use_search_point.z = searchPoint.z;
    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
    pi = rot * pi + t;
    use_search_point.x = pi[0];
    use_search_point.y = pi[1];
    use_search_point.z = pi[2];
    Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y,
                       searchPoint.normal_z);
    ni = rot * ni;
    if (kd_tree->nearestKSearch(use_search_point, 1, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      pcl::PointXYZINormal nearstPoint =
          target_cloud->points[pointIdxNKNSearch[0]];
      Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
      Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y,
                          nearstPoint.normal_z);
      Eigen::Vector3d normal_inc = ni - tni;
      Eigen::Vector3d normal_add = ni + tni;
      double point_to_plane = fabs(tni.transpose() * (pi - tpi));
      if ((normal_inc.norm() < normal_threshold ||
           normal_add.norm() < normal_threshold) &&
          point_to_plane < dis_threshold) {
        useful_match++;
      }
    }
  }
  return useful_match / source_cloud->size();
}

