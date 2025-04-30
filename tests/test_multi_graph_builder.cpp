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

void SetKnnParams(efanna2e::Parameters& paras, int baseid, int argc, char** argv) {
  unsigned K = (unsigned)atoi(argv[baseid + 0]);
  unsigned L = (unsigned)atoi(argv[baseid + 1]);
  unsigned iter = (unsigned)atoi(argv[baseid + 2]);
  unsigned S = (unsigned)atoi(argv[baseid + 3]);
  unsigned R = (unsigned)atoi(argv[baseid + 4]);

  paras.Set<unsigned>("K", K);
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("iter", iter);
  paras.Set<unsigned>("S", S);
  paras.Set<unsigned>("R", R);
}

void SetNsgParams(efanna2e::Parameters& paras, int baseid, int argc, char** argv) {
  unsigned nsg_L = (unsigned)atoi(argv[baseid]);
  unsigned nsg_R = (unsigned)atoi(argv[baseid + 1]);
  unsigned nsg_C = (unsigned)atoi(argv[baseid + 2]);
  unsigned search_L = (unsigned)atoi(argv[baseid + 3]);
  std::string performance_log(argv[baseid + 4]);
  paras.Set<unsigned>("L", nsg_L);
  paras.Set<unsigned>("R", nsg_R);
  paras.Set<unsigned>("C", nsg_C);
  paras.Set<std::string>("performance_log", performance_log);
  paras.Set<unsigned>("Search_L", search_L);
  std::cout << "  参数: L=" << nsg_L << ", R=" << nsg_R << ", C=" << nsg_C 
            << ", search_L=" << search_L << std::endl;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "Usage: "<< argv[0] << " data_file num_graph [nn_file K L iter S R nsg_L nsg_R nsg_C search_L performance_log nsg_graph] query_file groundtruth_file" << std::endl;
    exit(-1);
  }
  float* data_load = NULL;

  unsigned points_num, dim;
  load_data(argv[1], data_load, points_num, dim);
  int num_graphs = atoi(argv[2]);

  if (argc != 3 + num_graphs * 12 + 2) {
    std::cout << "Usage: "<< argv[0] << " data_file num_graph [nn_file K L iter S R nsg_L nsg_R nsg_C search_L performance_log nsg_graph] query_file groundtruth_file" << std::endl;
    std::cout << "argc: " << argc << std::endl;
    exit(-1);
  }
  efanna2e::MultiGraphBuilder builder(dim, points_num, efanna2e::L2);
  {
    for (int i = 0; i < num_graphs; i++){
      efanna2e::IndexRandom init_index(dim, points_num);
      efanna2e::IndexGraph index(dim, points_num, efanna2e::L2, (efanna2e::Index*)(&init_index));
      int base_idx = 3 + i * 12;
      char* graph_filename = argv[base_idx];
      efanna2e::Parameters knn_params, nsg_params;
      SetKnnParams(knn_params, base_idx + 1, argc, argv);
      SetNsgParams(nsg_params, base_idx + 6, argc, argv);
      std::string save_graph_path(argv[base_idx + 11]);
      index.Build(points_num, data_load, knn_params);
      index.Save(graph_filename);
      // using index.final_graph
      // no load knn
      builder.AddGraphConfig(nsg_params, save_graph_path, index.ExtractFinalGraph());
    }
  }
  auto s = std::chrono::high_resolution_clock::now();
  std::cout << std::endl << "开始构建 " << num_graphs << " 个图..." << std::endl;
  builder.BuildAllGraphs(data_load);
  
  auto e = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = e - s;
  
  std::cout << "多图构建总耗时: " << diff.count() << " 秒" << std::endl;
  //search
  {
    unsigned base_id = 3 + num_graphs * 12;
    float* query_data = nullptr;
    unsigned query_num;
    load_data(argv[base_id], query_data, query_num, dim);

    std::vector<std::vector<unsigned>> gtrue;
    load_ivecs(argv[base_id + 1], gtrue);
    // std::vector<unsigned> L_values = {60, 80, 100, 120, 140, 160, 180, 200};
    // builder.EvaluateGraphs(query_data, query_num, 100, L_values,
    //                       gtrue, argv[base_id + 2]);

    builder.EvaluateGraphs(query_data, query_num, 100, gtrue);
  }
  // delete[] data_load;
  return 0;
}