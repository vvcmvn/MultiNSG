//
// 多图构建测试程序
//

#include <efanna2e/multi_graph_builder.h>
#include <chrono>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <string>
#include <efanna2e/index_random.h>
#include <efanna2e/index_graph.h>

void load_data(const char* filename, float*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[(size_t)num * (size_t)dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}
void load_ivecs(const char* filename, std::vector<std::vector<unsigned>>& result){
  std::ifstream in(filename, std::ios::binary);
  if(!in.is_open()){
    std::cout << "open file error" << std::endl;
  }
  unsigned K;
  in.read((char*)&K, sizeof(unsigned));
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  unsigned num = (unsigned)(fsize / (K + 1) / 4);
  in.seekg(0, std::ios::beg);
  result.reserve(num);
  result.resize(num);
  unsigned kk = (K + 3) / 4 * 4;
  for(unsigned i = 0; i < num; i++){
    in.seekg(4, std::ios::cur);
    result[i].resize(K);
    result[i].reserve(kk);
    in.read((char*)result[i].data(), K * sizeof(unsigned));
  }
  in.close();
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "data_file num_graph [nn_file K L iter S R nsg_L nsg_R nsg_C nsg_graph] query_file groundtruth_file" << std::endl;
    exit(-1);
  }
  // 解析基本参数
  float* data_load = NULL;

  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);

  
  int num_graphs = atoi(argv[2]);

  // 检查参数数量是否足够
  if (argc != 3 + num_graphs * 10 + 3) {
    std::cout << "data_file num_graph [nn_file K L iter S R nsg_L nsg_R nsg_C nsg_graph] query_file groundtruth_file results_file" << std::endl;
    std::cout << "argc" << argc << std::endl;
    exit(-1);
  }
  // omp_set_num_threads(omp_get_max_threads()/2);
  // knn构图
  {
    for (int i = 0; i < num_graphs; i++){
      int base_idx = 3 + i * 10;
      char* graph_filename = argv[base_idx];
      unsigned K = (unsigned)atoi(argv[base_idx + 1]);
      unsigned L = (unsigned)atoi(argv[base_idx + 2]);
      unsigned iter = (unsigned)atoi(argv[base_idx + 3]);
      unsigned S = (unsigned)atoi(argv[base_idx + 4]);
      unsigned R = (unsigned)atoi(argv[base_idx + 5]);
      efanna2e::IndexRandom init_index(dim, points_num);
      efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index*)(&init_index));
      efanna2e::Parameters paras;
      paras.Set<unsigned>("K", K);
      paras.Set<unsigned>("L", L);
      paras.Set<unsigned>("iter", iter);
      paras.Set<unsigned>("S", S);
      paras.Set<unsigned>("R", R);
      index.Build(points_num, data_load, paras);
      index.Save(graph_filename);
    }
  }
  // 创建多图构建器
  efanna2e::MultiGraphBuilder builder(dim, points_num, efanna2e::L2);
  auto s = std::chrono::high_resolution_clock::now();
  // 添加图配置
  for (int i = 0; i < num_graphs; i++) {
    int base_idx = 3 + i * 10;
    
    std::string nn_graph_path(argv[base_idx]);
    unsigned L = (unsigned)atoi(argv[base_idx + 6]);
    unsigned R = (unsigned)atoi(argv[base_idx + 7]);
    unsigned C = (unsigned)atoi(argv[base_idx + 8]);
    std::string save_graph_path(argv[base_idx + 9]);
    
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L", L);
    paras.Set<unsigned>("R", R);
    paras.Set<unsigned>("C", C);
    
    builder.AddGraphConfig(paras, save_graph_path, nn_graph_path);
    
    std::cout << "添加图 #" << (i+1) << ":" << std::endl;
    std::cout << "  近邻图: " << nn_graph_path << std::endl;
    std::cout << "  参数: L=" << L << ", R=" << R << ", C=" << C << std::endl;
    std::cout << "  输出路径: " << save_graph_path << std::endl;
  }
  
  // 构建所有图
  std::cout << std::endl << "开始构建 " << num_graphs << " 个图..." << std::endl;
  builder.BuildAllGraphs(data_load);
  
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  
  std::cout << "多图构建总耗时: " << diff.count() << " 秒" << std::endl;
  //search
  {
    unsigned base_id = 3 + num_graphs * 10;
    float* query_data = nullptr;
    unsigned query_num;
    load_data(argv[base_id], query_data, query_num, dim);

    std::vector<std::vector<unsigned>> gtrue;
    load_ivecs(argv[base_id + 1], gtrue);
    std::vector<unsigned> L_values = {60, 80, 100, 120, 140, 160, 180, 200};
    builder.EvaluateGraphs(query_data, query_num, 100, L_values,
                          gtrue, argv[base_id + 2]);
  }
  // delete[] data_load;
  return 0;
}