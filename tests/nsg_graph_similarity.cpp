#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>

// NSG图相似度分析工具
class NSGGraphSimilarity {
public:
    using Graph = std::vector<std::vector<unsigned>>;
    
    // 从NSG二进制文件加载图
    bool LoadGraph(const std::string& filename, Graph& graph) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }
        
        // 读取图头信息
        unsigned width, ep;
        in.read((char*)&width, sizeof(unsigned));
        in.read((char*)&ep, sizeof(unsigned));
        
        // 读取图结构
        graph.clear();
        unsigned node_count = 0;
        while (!in.eof()) {
            unsigned k;
            in.read((char*)&k, sizeof(unsigned));
            if (in.eof()) break;
            std::vector<unsigned> neighbors(k);
            in.read((char*)neighbors.data(), k * sizeof(unsigned));
            graph.push_back(neighbors);
            node_count++;
        }
        
        std::cout << "从 " << filename << " 加载了 " << node_count << " 个节点" << std::endl;
        return true;
    }
    
    // 计算Jaccard相似度 (交集/并集)
    double CalculateJaccard(const std::vector<unsigned>& neighbors1, 
                         const std::vector<unsigned>& neighbors2) {
        if (neighbors1.empty() && neighbors2.empty()) return 1.0;
        
        std::unordered_set<unsigned> set1(neighbors1.begin(), neighbors1.end());
        unsigned intersection = 0;
        for (unsigned id : neighbors2) {
            if (set1.count(id) > 0) intersection++;
        }
        
        unsigned union_size = neighbors1.size() + neighbors2.size() - intersection;
        return static_cast<double>(intersection) / union_size;
    }
    
    // 计算重合系数 (交集/min(A,B))
    double CalculateOverlap(const std::vector<unsigned>& neighbors1, 
                         const std::vector<unsigned>& neighbors2) {
        if (neighbors1.empty() && neighbors2.empty()) return 1.0;
        if (neighbors1.empty() || neighbors2.empty()) return 0.0;
        
        std::unordered_set<unsigned> set1(neighbors1.begin(), neighbors1.end());
        unsigned intersection = 0;
        for (unsigned id : neighbors2) {
            if (set1.count(id) > 0) intersection++;
        }
        
        unsigned min_size = std::min(neighbors1.size(), neighbors2.size());
        return static_cast<double>(intersection) / min_size;
    }
    
    // 分析两个图的相似性
    void AnalyzeGraphSimilarity(const std::string& graph1_path,
                              const std::string& graph2_path) {
        Graph graph1, graph2;
        
        if (!LoadGraph(graph1_path, graph1) || !LoadGraph(graph2_path, graph2)) {
            std::cerr << "加载图失败" << std::endl;
            return;
        }
        
        // 确保节点数相同
        size_t n = std::min(graph1.size(), graph2.size());
        if (graph1.size() != graph2.size()) {
            std::cout << "警告: 图大小不一致. 仅比较前 " << n << " 个节点" << std::endl;
        }
        
        // 各种指标统计
        std::vector<double> jaccard_similarities(n);
        std::vector<double> overlap_coefficients(n);
        unsigned total_edges1 = 0, total_edges2 = 0, common_edges = 0;
        
        // 计算每个节点的相似度
        for (size_t i = 0; i < n; i++) {
            jaccard_similarities[i] = CalculateJaccard(graph1[i], graph2[i]);
            overlap_coefficients[i] = CalculateOverlap(graph1[i], graph2[i]);
            
            total_edges1 += graph1[i].size();
            total_edges2 += graph2[i].size();
            
            // 计算共同边
            std::unordered_set<unsigned> neighbors1(graph1[i].begin(), graph1[i].end());
            for (unsigned id : graph2[i]) {
                if (neighbors1.count(id) > 0) {
                    common_edges++;
                }
            }
            
            // 输出部分节点的详细信息
            // if (i < 5 || i == 100 || i == 500 || i == 1000 || i % 10000 == 0) {
            //     std::cout << "\n节点 " << i << ":" << std::endl;
            //     std::cout << "  Jaccard相似度: " << std::fixed << std::setprecision(4) 
            //               << jaccard_similarities[i] << std::endl;
            //     std::cout << "  重合系数: " << overlap_coefficients[i] << std::endl;
            //     std::cout << "  图1度数: " << graph1[i].size() 
            //               << ", 图2度数: " << graph2[i].size() << std::endl;
            // }
        }
        
        // 计算统计指标
        double avg_jaccard = std::accumulate(jaccard_similarities.begin(), 
                                          jaccard_similarities.end(), 0.0) / n;
        double avg_overlap = std::accumulate(overlap_coefficients.begin(), 
                                          overlap_coefficients.end(), 0.0) / n;
        
        // 排序计算中位数
        std::vector<double> sorted_jaccard = jaccard_similarities;
        std::sort(sorted_jaccard.begin(), sorted_jaccard.end());
        
        // 输出总体统计
        std::cout << "\n============ 图相似度分析结果 ============" << std::endl;
        std::cout << "节点数: " << n << std::endl;
        std::cout << "图1边数: " << total_edges1 << " (平均度数: " 
                  << static_cast<double>(total_edges1)/n << ")" << std::endl;
        std::cout << "图2边数: " << total_edges2 << " (平均度数: " 
                  << static_cast<double>(total_edges2)/n << ")" << std::endl;
        std::cout << "共同边数: " << common_edges << std::endl;
        
        // 计算边的Jaccard相似度
        double edge_jaccard = static_cast<double>(common_edges) / 
                            (total_edges1 + total_edges2 - common_edges);
        std::cout << "全图边的Jaccard相似度: " << edge_jaccard << std::endl;
        
        // 节点级别统计
        std::cout << "\n----- 节点级别相似度统计 -----" << std::endl;
        std::cout << "Jaccard相似度均值: " << avg_jaccard << std::endl;
        std::cout << "Jaccard相似度中位数: " << sorted_jaccard[n/2] << std::endl;
        std::cout << "重合系数均值: " << avg_overlap << std::endl;
        
        // 相似度分布
        OutputSimilarityDistribution(jaccard_similarities);
    }
    
    // 输出相似度分布
    void OutputSimilarityDistribution(const std::vector<double>& similarities) {
        std::cout << "\n----- 相似度分布 -----" << std::endl;
        int bins[10] = {0};
        for (double sim : similarities) {
            int bin = std::min(static_cast<int>(sim * 10), 9);
            bins[bin]++;
        }
        
        for (int i = 0; i < 10; i++) {
            double start = i / 10.0;
            double end = (i + 1) / 10.0;
            std::cout << "[" << std::fixed << std::setprecision(1) 
                      << start << "-" << end << "): " 
                      << bins[i] << " 个节点 (" 
                      << std::setprecision(2) 
                      << 100.0 * bins[i] / similarities.size() << "%)" 
                      << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " <图1路径> <图2路径>" << std::endl;
        return 1;
    }
    
    NSGGraphSimilarity analyzer;
    analyzer.AnalyzeGraphSimilarity(argv[1], argv[2]);
    
    return 0;
}