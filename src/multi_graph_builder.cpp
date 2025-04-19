#include "efanna2e/multi_graph_builder.h"
#include <omp.h>
#include <chrono>
#include <unordered_set>
#include <cmath>
#include <stack>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#define BYPASS_CACHE 0
#define REPEAT_COUNT 5

namespace efanna2e {

MultiGraphBuilder::MultiGraphBuilder(const size_t dimension, const size_t n, Metric m)
    : dimension_(dimension), n_(n), metric_(m) {
    switch (m) {
      case L2:
        distance_ = new DistanceL2();
        break;
      case INNER_PRODUCT:
        distance_ = new DistanceInnerProduct();
        break;
      default:
        distance_ = new DistanceL2();
        break;
    }
}

MultiGraphBuilder::~MultiGraphBuilder() {
    if (distance_ != nullptr) {
        delete distance_;
        distance_ = nullptr;
    }
    
    // 清理任何未释放的cut_graph
    for (auto& config : graph_configs_) {
        if (config.cut_graph != nullptr) {
            delete[] config.cut_graph;
            config.cut_graph = nullptr;
        }
    }
}

void MultiGraphBuilder::AddGraphConfig(const Parameters& parameters, const std::string& save_path, 
                                     const std::string& nn_graph_path) {
    GraphConfig config;
    config.parameters = parameters;
    config.save_path = save_path;
    config.nn_graph_path = nn_graph_path;
    config.width = parameters.Get<unsigned>("R");
    graph_configs_.push_back(config);
}

float MultiGraphBuilder::GetPointDistance(unsigned query_id, unsigned other_id, 
                                        std::vector<float>& point_distances,
                                        boost::dynamic_bitset<>& computed_flags) {

    #if BYPASS_CACHE
        return distance_->compare(
        data_ + dimension_ * (size_t)query_id,
        data_ + dimension_ * (size_t)other_id,
        dimension_);
    #else
        if (computed_flags[other_id]) {
            return point_distances[other_id];
        }
        // 否则计算距离并缓存
        float dist = distance_->compare(
            data_ + dimension_ * (size_t)query_id,
            data_ + dimension_ * (size_t)other_id,
            dimension_);
        point_distances[other_id] = dist;
        computed_flags.set(other_id);
        return dist;
    #endif
}

float MultiGraphBuilder::GetPointDistance(unsigned query_id, unsigned other_id, std::vector<float>& point_distances) {
    if(point_distances[other_id] != -1){
        return point_distances[other_id];
    }
    float dist = distance_->compare(data_ + dimension_ * (size_t)query_id, data_ + dimension_ * (size_t)other_id, (unsigned)dimension_);
    point_distances[other_id] = dist;
    return dist;
}

void MultiGraphBuilder::LoadNNGraph(const std::string& filename, std::vector<std::vector<unsigned>>& graph) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        exit(-1);
    }
    unsigned k;
    in.read((char*)&k, sizeof(unsigned));
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    size_t num = (unsigned)(fsize / (k + 1) / 4);
    in.seekg(0, std::ios::beg);
    graph.resize(num);
    graph.reserve(num);
    unsigned kk = (k + 3) / 4 * 4;
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        graph[i].resize(k);
        graph[i].reserve(kk);
        in.read((char*)graph[i].data(), k * sizeof(unsigned));
    }
    in.close();
}
//todos
void MultiGraphBuilder::FindNeighbors(const float* query, const Parameters &parameter,
                                    std::vector<Neighbor>& retset,
                                    std::vector<Neighbor>& fullset,
                                    unsigned& ep,
                                    const std::vector<std::vector<unsigned>>& graph){
    unsigned L = parameter.Get<unsigned>("L");

    retset.resize(L + 1);
    std::vector<unsigned> init_ids(L);
    // initializer_->Search(query, nullptr, L, parameter, init_ids.data());

    boost::dynamic_bitset<> flags{n_, 0};
    L = 0;
    for (unsigned i = 0; i < init_ids.size() && i < graph[ep].size(); i++) {
        init_ids[i] = graph[ep][i];
        flags[init_ids[i]] = true;
        L++;
    }
    while (L < init_ids.size()) {
        unsigned id = rand() % n_;
        if (flags[id]) continue;
        init_ids[L] = id;
        L++;
        flags[id] = true;
    }
    //计算query和init_ids中点的距离
    L = 0;
    for (unsigned i = 0; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        if (id >= n_) continue;
        // std::cout<<id<<std::endl;
        float dist = distance_->compare(data_ + dimension_ * (size_t)id, query,
                                        (unsigned)dimension_);
        retset[i] = Neighbor(id, dist, true);
        // flags[id] = 1;
        L++;
    }

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < (int)L) {
    int nk = L;

    if (retset[k].flag) {
    retset[k].flag = false;
    unsigned n = retset[k].id;
    for (unsigned m = 0; m < graph[n].size(); ++m) {
        unsigned id = graph[n][m];
        if (flags[id]) continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * (size_t)id,
                                        (unsigned)dimension_);
        Neighbor nn(id, dist, true);
        fullset.push_back(nn);
        if (dist >= retset[L - 1].distance) continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if (L + 1 < retset.size()) ++L;
        if (r < nk) nk = r;
    }
    }
    if (nk <= k)
        k = nk;
    else
        ++k;
    }
}
void MultiGraphBuilder::FindNeighbors(unsigned query_id, 
                                    const Parameters& parameter,
                                    boost::dynamic_bitset<>& flags,
                                    std::vector<Neighbor>& retset,
                                    std::vector<Neighbor>& fullset,
                                    const std::vector<std::vector<unsigned>>& graph,
                                    unsigned ep,
                                    std::vector<float>& point_distances,
                                    boost::dynamic_bitset<>& computed_flags) {
    unsigned L = parameter.Get<unsigned>("L");
    
    retset.resize(L + 1);
    std::vector<unsigned> init_ids(L);
    
    L = 0;
    for (unsigned i = 0; i < init_ids.size() && i < graph[ep].size(); i++) {
        init_ids[i] = graph[ep][i];
        flags[init_ids[i]] = true;
        L++;
    }
    
    // 不足L个，随机补充
    while (L < init_ids.size()) {
        unsigned id = rand() % n_;
        if (flags[id]) continue;
        init_ids[L] = id;
        flags[id] = true;
        L++;
    }
    
    // 计算距离
    L = 0;
    for (unsigned i = 0; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        if (id >= n_) continue;
        // test
        float dist = GetPointDistance(query_id, id, point_distances, computed_flags);
        // float dist = distance_->compare(data_ + dimension_ * (size_t)id, data_ + dimension_ * (size_t)query_id, (unsigned)dimension_);
        retset[i] = Neighbor(id, dist, true);
        fullset.push_back(retset[i]);
        L++;
    }
    
    // 排序并搜索
    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < (int)L) {
        int nk = L;
        
        if (retset[k].flag) {
            retset[k].flag = false;
            unsigned n = retset[k].id;
            
            for (unsigned m = 0; m < graph[n].size(); m++) {
                unsigned id = graph[n][m];
                if (flags[id]) continue;
                flags[id] = true;
                //test
                float dist = GetPointDistance(query_id, id, point_distances, computed_flags);
                // float dist = distance_->compare(data_ + dimension_ * (size_t)id, data_ + dimension_ * (size_t)query_id, (unsigned)dimension_);
                Neighbor nn(id, dist, true);
                fullset.push_back(nn);
                
                // 检查是否加入结果集
                if (dist >= retset[L - 1].distance) continue;
                int r = InsertIntoPool(retset.data(), L, nn);
                
                if (L + 1 < retset.size()) ++L;
                if (r < nk) nk = r;
            }
        }
        
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
}

void MultiGraphBuilder::PruneNeighbors(unsigned q, 
                                     std::vector<Neighbor>& pool,
                                     const Parameters& parameter,
                                     boost::dynamic_bitset<>& flags,
                                     SimpleNeighbor* cut_graph,
                                     std::vector<float>& point_distances,
                                     boost::dynamic_bitset<>& computed_flags,
                                     CompactGraph& graph) {
    unsigned range = parameter.Get<unsigned>("R");
    unsigned maxc = parameter.Get<unsigned>("C");
    unsigned start = 0;

    for (unsigned nn = 0; nn < graph[q].size(); nn++) {
        unsigned id = graph[q][nn];
        if (flags[id]) continue;
        float dist = GetPointDistance(q,
                                    id, 
                                    point_distances,
                                    computed_flags);
        // float dist = distance_->compare(data_ + dimension_ * (size_t)q,
        //                                 data_ + dimension_ * (size_t)id, 
        //                                 (unsigned)dimension_);
        pool.push_back(Neighbor(id, dist, true));
    }
    std::sort(pool.begin(), pool.end());
    // if (q == 100 || q == 500 || q == 1000) { // 选择一些代表性节点
    //     std::cout << "-----Processing node " << q << "-----" << std::endl;
    //     std::cout << "Pool size before pruning: " << pool.size() << std::endl;
        
    //     // 输出初始池中的一些点
    //     for (size_t i = 0; i < std::min(pool.size(), size_t(10)); i++) {
    //         std::cout << "Pool[" << i << "]: id=" << pool[i].id 
    //                     << ", dist=" << pool[i].distance << std::endl;
    //     }
        
    //     // 创建日志文件记录完整过程
    //     std::ofstream log("origin_node_"+std::to_string(q)+"_build.log");
    //     for (const auto& p : pool) {
    //         log << p.id << " " << p.distance << std::endl;
    //     }
    //     log.close();
    // }
    std::vector<Neighbor> result;
    if (pool[start].id == q) start++;
    result.push_back(pool[start]);
    // 剪枝
    while (result.size() < range && (++start) < pool.size() && start < maxc) {
        auto& p = pool[start];
        bool occlude = false;
        for (unsigned t = 0; t < result.size(); t++) {
            if (p.id == result[t].id) {
                occlude = true;
                break;
            }
            float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id, data_ + dimension_ * (size_t)p.id, (unsigned)dimension_);
            if (djk < p.distance) {
                occlude = true;
                break;
            }
        }
        if (!occlude) result.push_back(p);
    }
    // 保存结果
    SimpleNeighbor* des_pool = cut_graph + (size_t)q * (size_t)range;
    for (size_t t = 0; t < result.size(); t++) {
        des_pool[t].id = result[t].id;
        des_pool[t].distance = result[t].distance;
    }
    
    if (result.size() < range) {
        des_pool[result.size()].distance = -1;
    }
}
    //待检查
void MultiGraphBuilder::InterInsert(unsigned n, 
                                  unsigned range,
                                  std::vector<std::mutex>& locks,
                                  SimpleNeighbor* cut_graph) {
    SimpleNeighbor* src_pool = cut_graph + (size_t)n * (size_t)range;
    
    for (size_t i = 0; i < range; i++) {
        if (src_pool[i].distance == -1) break;
        
        SimpleNeighbor sn(n, src_pool[i].distance);
        size_t des = src_pool[i].id;
        SimpleNeighbor* des_pool = cut_graph + des * (size_t)range;
        
        std::vector<SimpleNeighbor> temp_pool;
        int dup = 0;
        
        {
            LockGuard guard(locks[des]);
            // std::lock_guard<std::mutex> guard(locks[des]);
            for (size_t j = 0; j < range; j++) {
                if (des_pool[j].distance == -1) break;
                if (n == des_pool[j].id) {
                    dup = 1;
                    break;
                }
                temp_pool.push_back(des_pool[j]);
            }
        }
        if (dup) continue;
        temp_pool.push_back(sn);
        
        if (temp_pool.size() > range) {
            std::vector<SimpleNeighbor> result;
            unsigned start = 0;
            
            std::sort(temp_pool.begin(), temp_pool.end());
            result.push_back(temp_pool[start]);
            
            // 剪枝
            while (result.size() < range && (++start) < temp_pool.size()) {
                auto& p = temp_pool[start];
                bool occlude = false;
                
                for (unsigned t = 0; t < result.size(); t++) {
                    if (p.id == result[t].id) {
                        occlude = true;
                        break;
                    }
                    
                    float djk = distance_->compare(
                        data_ + dimension_ * (size_t)result[t].id,
                        data_ + dimension_ * (size_t)p.id,
                        (unsigned)dimension_
                    );
                    
                    if (djk < p.distance) {
                        occlude = true;
                        break;
                    }
                }
                
                if (!occlude) result.push_back(p);
            }
            {
                LockGuard guard(locks[des]);
                // std::lock_guard<std::mutex> guard(locks[des]);
                for (unsigned t = 0; t < result.size(); t++) {
                    des_pool[t] = result[t];
                }
            }
        } else {
            LockGuard guard(locks[des]);
            // std::lock_guard<std::mutex> guard(locks[des]);
            for (unsigned t = 0; t < range; t++) {
                if (des_pool[t].distance == -1) {
                    des_pool[t] = sn;
                    if (t + 1 < range) des_pool[t + 1].distance = -1;
                    break;
                }
            }
        }
    }
}

void MultiGraphBuilder::InitGraph(const Parameters& parameters, 
                                std::vector<std::vector<unsigned>>& final_graph, 
                                unsigned& ep) {
    // 计算数据集中心点
    float* center = new float[dimension_];
    for (unsigned j = 0; j < dimension_; j++) center[j] = 0;
    for (unsigned i = 0; i < n_; i++) {
        for (unsigned j = 0; j < dimension_; j++) {
            center[j] += data_[i * dimension_ + j];
        }
    }
    for (unsigned j = 0; j < dimension_; j++) {
        center[j] /= n_;
    }
    std::vector<Neighbor> tmp, pool;
    // 找到最近邻作为入口点
    ep = rand() % n_;
    FindNeighbors(center, parameters, tmp, pool, ep, final_graph);
    ep = tmp[0].id;
    delete[] center;
}

void MultiGraphBuilder::DFS(boost::dynamic_bitset<>& flag, unsigned root, unsigned& cnt,
    const std::vector<std::vector<unsigned>>& final_graph) {
    unsigned tmp = root;
    std::stack<unsigned> s;
    s.push(root);
    if (!flag[root]) cnt++;
    flag[root] = true;
    while (!s.empty()) {
        unsigned next = n_ + 1;
        for (unsigned i = 0; i < final_graph[tmp].size(); i++) {
            if (flag[final_graph[tmp][i]] == false) {
                next = final_graph[tmp][i];
                break;
            }
        }
        // std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
        if (next == (n_ + 1)) {
            s.pop();
            if (s.empty()) break;
            tmp = s.top();
            continue;
        }
            tmp = next;
            flag[tmp] = true;
            s.push(tmp);
            cnt++;
    }
}

void MultiGraphBuilder::FindRoot(boost::dynamic_bitset<>& flag, unsigned& root,
         const Parameters& parameter,
         std::vector<std::vector<unsigned>>& final_graph) {
// 查找第一个未连接的点
    unsigned id = n_;
    for (unsigned i = 0; i < n_; i++) {
    if (flag[i] == false) {
        id = i;
        break;
    }
    }

    if (id == n_) return;  // No Unlinked Node

    std::vector<Neighbor> tmp, pool;
    FindNeighbors(data_ + dimension_ * id, parameter, tmp, pool, root, final_graph);
    std::sort(pool.begin(), pool.end());

    unsigned found = 0;
    for (unsigned i = 0; i < pool.size(); i++) {
    if (flag[pool[i].id]) {
        // std::cout << pool[i].id << '\n';
        root = pool[i].id;
        found = 1;
        break;
    }
    }
    if (found == 0) {
    while (true) {
        unsigned rid = rand() % n_;
        if (flag[rid]) {
        root = rid;
        break;
        }
    }
    }
    final_graph[root].push_back(id);
}

void MultiGraphBuilder::TreeGrow(const Parameters& parameter,
          std::vector<std::vector<unsigned>>& final_graph,
          unsigned& ep,
          unsigned& width) {
    boost::dynamic_bitset<> flags{n_, 0};
    unsigned unlinked_cnt = 0;
    unsigned root = ep;

    while (unlinked_cnt < n_) {
        DFS(flags, root, unlinked_cnt, final_graph);
        if (unlinked_cnt >= n_) break;
        FindRoot(flags, root, parameter, final_graph);
    }
    // 更新图的最大宽度
    width = 0;
    for (size_t i = 0; i < n_; i++) {
        if (final_graph[i].size() > width) {
            width = final_graph[i].size();
        }
    }
}

void MultiGraphBuilder::ConvertToFinalGraph(GraphConfig& config) {
    unsigned range = config.parameters.Get<unsigned>("R");
    
    // 从临时图结构转换为最终图
    // config.final_graph.resize(n_);
    for (size_t i = 0; i < n_; i++) {
        SimpleNeighbor* pool = config.cut_graph + i * (size_t)range;
        
        unsigned pool_size = 0;
        for (unsigned j = 0; j < range; j++) {
            if (pool[j].distance == -1) break;
            pool_size = j;
        }
        pool_size++;
        
        config.final_graph[i].resize(pool_size);
        for (unsigned j = 0; j < pool_size; j++) {
            config.final_graph[i][j] = pool[j].id;
        }
    }
}



void MultiGraphBuilder::BuildAllGraphs(const float* data) {
    data_ = data;
    // 加载所有近邻图
    srand(42);  // 使用固定种子42
    for (auto& config : graph_configs_) {
        std::cout << "Loading nn_graph from: " << config.nn_graph_path << std::endl;
        LoadNNGraph(config.nn_graph_path, config.final_graph);

        // 初始化图结构
        config.final_graph.resize(n_);
        InitGraph(config.parameters, config.final_graph, config.ep);

        // 为每个图分配临时图结构
        unsigned range = config.parameters.Get<unsigned>("R");
        config.cut_graph = new SimpleNeighbor[n_ * (size_t)range];
    }
    std::cout << "Building all graphs..." << std::endl;
    // double total_time = 0.0;
    // double total_compute_time = 0.0;
    // double time = 0.0;
    // double total_prune_time = 0.0;
    // double total_get_neighbors_time = 0.0;
    // auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        std::vector<float> point_distances(n_, 0);
        std::vector<Neighbor> pool, tmp;
        boost::dynamic_bitset<> flags{n_, 0};
        boost::dynamic_bitset<> computed_flags(n_,0); 
        std::unordered_map<unsigned, float> point_neighbors;
        // auto thread_time = 0.0;
        // auto thread_get_neighbors_time = 0.0;
        // auto thread_prune_time = 0.0;
        // auto thread_compute_time = 0.0;
        #pragma omp for schedule(dynamic, 100)
        for (unsigned n = 0; n < n_; ++n) {
            computed_flags.reset();
            for (auto& config : graph_configs_) {
                pool.clear();
                tmp.clear();
                flags.reset();
                // auto start_time = std::chrono::high_resolution_clock::now();
                FindNeighbors(n, config.parameters, flags, tmp, pool, 
                           config.final_graph, config.ep, point_distances, computed_flags);
                // auto end_time = std::chrono::high_resolution_clock::now();
                PruneNeighbors(n, pool, config.parameters, flags, 
                             config.cut_graph,point_distances,computed_flags,config.final_graph);
                // auto end_prune_time = std::chrono::high_resolution_clock::now();
                // thread_get_neighbors_time += std::chrono::duration<double>(end_time - start_time).count();
                // thread_prune_time += std::chrono::duration<double>(end_prune_time - end_time).count();
                // thread_compute_time += std::chrono::duration<double>(end_prune_time - start_time).count();
                // thread_time += std::chrono::duration<double>(end_prune_time - start_time).count();
            }
        }
        // #pragma omp atomic
        // total_time += thread_time;
        // #pragma omp atomic
        // total_compute_time += thread_compute_time;
        // #pragma omp atomic
        // total_prune_time += thread_prune_time;
        // #pragma omp atomic
        // total_get_neighbors_time += thread_get_neighbors_time;
    }
    // auto end = std::chrono::high_resolution_clock::now();
    // time = std::chrono::duration<double>(end - start).count();
    // std::cout << "Construction time: " << time << std::endl;
    // std::cout << "computational time: " << total_compute_time << std::endl;
    // std::cout << "Percentage of time on computation: " << total_compute_time / total_time << std::endl;
    // std::cout << "prune time: " << total_prune_time << std::endl;
    // std::cout << "Percentage of time on prune: " << total_prune_time / total_time << std::endl;
    // std::cout << "get neighbors time: " << total_get_neighbors_time << std::endl;
    // std::cout << "Percentage of time on get neighbors: " << total_get_neighbors_time / total_time << std::endl;
    
    // 对每个图独立处理互连等后续步骤
    for (auto& config : graph_configs_) {
        unsigned range = config.parameters.Get<unsigned>("R");
        std::vector<std::mutex> locks(n_);
        // 反向边
        #pragma omp parallel for schedule(dynamic, 100)
        for (unsigned n = 0; n < n_; ++n) {
            InterInsert(n, range, locks, config.cut_graph);
        }
       
        // 转换为最终图格式
        ConvertToFinalGraph(config);
        TreeGrow(config.parameters, config.final_graph, config.ep, config.width);
        
        // 保存图
        std::ofstream out(config.save_path, std::ios::binary | std::ios::out);
        out.write((char*)&config.width, sizeof(unsigned));
        out.write((char*)&config.ep, sizeof(unsigned));
        
        for (unsigned i = 0; i < n_; i++) {
            unsigned GK = (unsigned)config.final_graph[i].size();
            out.write((char*)&GK, sizeof(unsigned));
            out.write((char*)config.final_graph[i].data(), GK * sizeof(unsigned));
        }
        out.close();
        
        // 统计图的度分布
        // unsigned max_degree = 0, min_degree = n_, avg_degree = 0;
        // for (size_t i = 0; i < n_; i++) {
        //     unsigned degree = config.final_graph[i].size();
        //     max_degree = std::max(max_degree, degree);
        //     min_degree = std::min(min_degree, degree);
        //     avg_degree += degree;
        // }
        // avg_degree /= n_;
        
        // std::cout << "Graph " << config.save_path << " built: " << std::endl;
        // std::cout << "  Max degree: " << max_degree << std::endl;
        // std::cout << "  Min degree: " << min_degree << std::endl;
        // std::cout << "  Avg degree: " << avg_degree << std::endl;
        
        // 释放资源
        delete[] config.cut_graph;
        config.cut_graph = nullptr;
    }
}

void MultiGraphBuilder::Search(const float* query, const size_t K, const size_t L, const GraphConfig& graph_config, unsigned* indices){
    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    boost::dynamic_bitset<> flags{graph_config.final_graph.size(), 0};

    const CompactGraph& final_graph = graph_config.final_graph;
    const unsigned ep = graph_config.ep;
    unsigned tmp_l = 0;
    for(; tmp_l < L && tmp_l < final_graph[ep].size(); tmp_l++){
        init_ids[tmp_l] = final_graph[ep][tmp_l];
        flags[init_ids[tmp_l]] = true;
    }
    while(tmp_l < L){
        unsigned id = rand() % n_;
        if(flags[id]) continue;
        init_ids[tmp_l] = id;
        flags[id] = true;
        tmp_l++;
    }

    for(unsigned i = 0; i < init_ids.size(); i++){
        unsigned id = init_ids[i];
        float dist = distance_->compare(data_ + dimension_ * (size_t)id, query, (unsigned)dimension_);
        retset[i] = Neighbor(id, dist, true);
    }

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;

    while(k < (int)L){
        int nk = L;
        if(retset[k].flag){
            retset[k].flag = false;
            unsigned n = retset[k].id;

            for(unsigned m = 0; m < final_graph[n].size(); ++m){
                unsigned id = final_graph[n][m];
                if(flags[id]) continue;
                flags[id] = 1;
                float dist = distance_->compare(query, data_ + dimension_ * (size_t)id, (unsigned)dimension_);
                if(dist >= retset[L - 1].distance) continue;
                Neighbor nn(id, dist, true);
                int r = InsertIntoPool(retset.data(), L, nn);

                if (r < nk) nk = r;
            }
        }
        if(nk <= k){
            k = nk;
        }
        else ++k;
    }
    for (size_t i = 0; i < K; i++){
        indices[i] = retset[i].id;
    }
}

void MultiGraphBuilder::EvaluateGraphs(const float* query_data, const size_t query_num, const size_t K, const std::vector<unsigned>& L_values,
                                    const std::vector<std::vector<unsigned>>& gtrue, const char* output_file){
    std::vector<GraphEvalResult> results;
    std::string structure_filename = std::string(output_file) + "_structure.csv";
    std::string performance_filename = std::string(output_file) + "_performance.csv";
    std::ofstream structure_out(structure_filename);
    std::ofstream performance_out(performance_filename);
    double total_time = 0.0;
    unsigned i = 0;
    for(auto& config : graph_configs_){
        std::vector<std::vector<unsigned>> search_result(query_num, std::vector<unsigned>(K));
        std::vector<double> run_times(REPEAT_COUNT);
        std::string graph_id = "graph" + std::to_string(i);
        {
            //CalculateAverageDegree & CalculateDegreeVariance
            boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::mean, boost::accumulators::tag::variance>> acc;
            double deg_variance = 0.0, avg_degree = 0.0;
            unsigned max_degree = 0;
            for (size_t i = 0; i < n_; i++) {
                unsigned degree = config.final_graph[i].size();
                acc(degree);
                max_degree = std::max(max_degree, degree);
            }
            avg_degree = boost::accumulators::mean(acc);
            deg_variance = boost::accumulators::variance(acc);
            structure_out << graph_id << ","
                    << avg_degree << "," 
                    << deg_variance << ","
                    << max_degree << std::endl;
        }
        
        for(unsigned L : L_values){
            total_time = 0.0;
            for(size_t j = 0; j < REPEAT_COUNT; j++){
                auto perf_start = std::chrono::high_resolution_clock::now();
                for(size_t i = 0; i < query_num; i++){
                    std::vector<unsigned> tmp(K);
                    Search(query_data + i * dimension_, K, L, config, search_result[i].data());
                }
                auto perf_end = std::chrono::high_resolution_clock::now();
                run_times[j] = (std::chrono::duration<double>(perf_end - perf_start).count());
                // std::cout << "search_L: " << L << " run: " << j << " time: " << run_times[j] << std::endl;
            }
            //skip the first run
            for(int run = 1; run < REPEAT_COUNT; run++){
                total_time += run_times[run];
            }
            double qps = query_num / (total_time / (REPEAT_COUNT - 1));
            float recall = ComputeRecall(gtrue, search_result, K);
            results.push_back(GraphEvalResult{graph_id, L, recall, qps});
        }
        i++;
    }
    for (const auto& res : results) {
        performance_out << res.graph_id << "," 
                   << res.search_L << "," 
                   << res.recall << "," 
                   << res.qps << std::endl;
    }
    structure_out.close();
    performance_out.close();
}

float MultiGraphBuilder::ComputeRecall(const std::vector<std::vector<unsigned>>& gtrue,
                                    const std::vector<std::vector<unsigned>>& results,
                                    size_t K) {
    unsigned query_num = results.size();
    
    float total_recall = 0.0;
    for(unsigned i = 0; i < query_num; i++){
        unsigned matches = 0;
        std::unordered_set<unsigned> gtrue_set(gtrue[i].begin(), gtrue[i].begin() + K);
        for(size_t j = 0; j < K; j++){
            if(gtrue_set.find(results[i][j]) != gtrue_set.end()){
                matches++;
            }
        }
        total_recall += static_cast<float>(matches) / K;
    }
    return query_num > 0 ? total_recall / query_num : 0.0f;
}
} // namespace efanna2e