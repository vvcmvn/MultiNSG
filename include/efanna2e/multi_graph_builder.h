#ifndef EFANNA2E_MULTI_GRAPH_BUILDER_H
#define EFANNA2E_MULTI_GRAPH_BUILDER_H

#include "index_nsg.h"
#include <vector>
#include <mutex>
#include <memory>
#include <string>
#include <boost/dynamic_bitset.hpp>
#include <atomic>

namespace efanna2e {

class MultiGraphBuilder {
public:
    typedef std::vector<std::vector<unsigned>> CompactGraph;
    MultiGraphBuilder(const size_t dimension, const size_t n, Metric m);
    ~MultiGraphBuilder();
    
    // 每个图使用不同的近邻图
    void AddGraphConfig(const Parameters &parameters, const std::string &save_path,
                        const std::string &nn_graph_path);
    void AddGraphConfig(const Parameters &parameters, const std::string &save_path,
                        const CompactGraph &&final_graph);
    // 构建所有图
    void BuildAllGraphs(const float *data);

    void EvaluateGraphs(const float *query_data, const size_t query_num, const size_t K,
                        const std::vector<unsigned> &L_values, const std::vector<std::vector<unsigned>> &gtrue, const char *output_file);
    void EvaluateGraphs(const float *query_data, const size_t query_num, const size_t K,
                        const std::vector<std::vector<unsigned>> &gtrue);

private:
    // 计算并缓存查询点的距离

    struct GraphConfig;
    void Search(const float *query, const size_t K, const size_t L, const GraphConfig &graph_config, unsigned *indices);
    float GetPointDistance(unsigned query_id, unsigned other_id, std::vector<float> &point_distances,
                           boost::dynamic_bitset<> &computed_flags);

    float GetPointDistance(unsigned query_id, unsigned other_id, std::vector<float>& point_distances);
    void DFS(boost::dynamic_bitset<>& flag, 
        unsigned root, 
        unsigned& cnt,
        const std::vector<std::vector<unsigned>>& final_graph);

// 寻找合适的连接点，用于连接孤立部分
    void FindRoot(boost::dynamic_bitset<>& flag, 
                unsigned& root,
                const Parameters& parameter,
                std::vector<std::vector<unsigned>>& final_graph);
    // 获取指定点在指定图中的邻居
    void FindNeighbors(const float* query, const Parameters &parameter,
                    std::vector<Neighbor>& retset,
                    std::vector<Neighbor>& fullset,
                    unsigned& ep,
                    const std::vector<std::vector<unsigned>>& graph);
    void FindNeighbors(unsigned query_id, 
                     const Parameters& parameter,
                     boost::dynamic_bitset<>& flags,
                     std::vector<Neighbor>& retset,
                     std::vector<Neighbor>& fullset,
                     const std::vector<std::vector<unsigned>>& graph,
                     unsigned ep,
                     std::vector<float>& point_distances,
                     boost::dynamic_bitset<>& computed_flags);
    void FindNeighbors(unsigned query_id, 
                    const Parameters& parameter,
                    boost::dynamic_bitset<>& flags,
                    std::vector<Neighbor>& retset,
                    std::vector<Neighbor>& fullset,
                    const std::vector<std::vector<unsigned>>& graph,
                    unsigned ep,
                    std::vector<float>& point_distances);
    
    // 对指定点的邻居集在指定图中进行剪枝
    void PruneNeighbors(unsigned q, 
                    std::vector<Neighbor>& pool,
                    const Parameters& parameter,
                    boost::dynamic_bitset<>& flags,
                    SimpleNeighbor* cut_graph,
                    std::vector<float>& point_distances,
                    boost::dynamic_bitset<>& computed_flags,
                    CompactGraph& graph);
    void PruneNeighbors(unsigned q, 
                    std::vector<Neighbor>& pool,
                    const Parameters& parameter,
                    boost::dynamic_bitset<>& flags,
                    SimpleNeighbor* cut_graph,
                    std::vector<float>& point_distances,
                    CompactGraph& graph);
                        
    // 互插入(每个图独立调用)
    void InterInsert(unsigned n, 
                   unsigned range,
                   std::vector<std::mutex>& locks,
                   SimpleNeighbor* cut_graph);
    
    // 初始化图(每个图独立调用)
    void InitGraph(const Parameters& parameters, 
                 std::vector<std::vector<unsigned>>& final_graph, 
                 unsigned& ep);
    
    // 加载近邻图
    void LoadNNGraph(const std::string& filename, std::vector<std::vector<unsigned>>& graph);

    void TreeGrow(const Parameters& parameter,
             std::vector<std::vector<unsigned>>& final_graph,
             unsigned& ep,
             unsigned& width);
    

    void ConvertToFinalGraph(GraphConfig& config);
    void SaveCutGraph(const std::string& filename, const SimpleNeighbor* cut_graph, 
        size_t n, unsigned range);
    
    void SaveFinalGraph(const std::string& filename, const CompactGraph& final_graph);
    float ComputeRecall(const std::vector<std::vector<unsigned>>& gtrue, const std::vector<std::vector<unsigned>>& results, size_t K);

    // 图配置结构
    
    struct GraphEvalResult {
        std::string graph_id;
        unsigned search_L;
        float recall;
        double qps;
        double average_degree;
        double degree_variance;         
        double distance_1tok_2;          
        
        GraphEvalResult(const std::string& id, unsigned L, float r, double q,
                       double avg_deg = 0.0, double deg_var = 0.0, double distance_1tok_2 = 0.0)
            : graph_id(id), search_L(L), recall(r), qps(q),
              average_degree(avg_deg), degree_variance(deg_var), distance_1tok_2(distance_1tok_2) {}
    };
    struct GraphConfig {
        Parameters parameters;
        std::string save_path;
        std::string nn_graph_path;    
        CompactGraph final_graph;//knn图
        SimpleNeighbor* cut_graph = nullptr;  
        unsigned ep;                   
        unsigned width = 0;          
    };
    // 基本属性
    const size_t dimension_;
    const size_t n_;
    Metric metric_;
    Distance* distance_;
    const float* data_{nullptr};
    
    // 图配置列表
    std::vector<GraphConfig> graph_configs_;
    std::vector<GraphEvalResult> eval_results_;
};

}  // namespace efanna2e

#endif  // EFANNA2E_MULTI_GRAPH_BUILDER_H