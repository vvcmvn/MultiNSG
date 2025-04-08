#!/bin/bash
# filepath: /home/xu/dev/nsg/scripts/run_multi_graph_builder.sh

# 默认参数
DATA_FILE="/home/xuzf/dev/ddpg_tuning/dataset/sift/sift_base.fvecs"

NUM_GRAPHS=1

#默认图配置 (每个数组元素包含5个参数: nn_graph L R C output)

CONFIGS=(
        "../nn_graphs/sift_nn.graph1 40 50 500 ../output/nsg_graph1"
)

# 显示使用帮助
show_help() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -h, --help                 显示此帮助信息"
    echo "  -d, --data FILE            指定数据文件 (默认: $DATA_FILE)"
    echo "  -n, --num-graphs NUMBER    指定构建图的数量 (默认: $NUM_GRAPHS)"
    echo "  -c, --config \"CONFIG\"      添加/替换图配置，格式: \"nn_graph L R C output\""
    echo "                             可多次使用此选项添加多个配置"
    echo "  --reset-configs            清除默认配置，仅使用通过-c提供的配置"
    echo
    echo "示例:"
    echo "  $0 -d ./my_data.bin"
    echo "  $0 -n 2 -c \"./my_graph.bin 200 100 500 ./my_output.bin\""
    echo "  $0 --reset-configs -c \"./g1.bin 100 50 300 ./o1.bin\" -c \"./g2.bin 200 100 500 ./o2.bin\""
}

# 解析命令行参数
RESET_CONFIGS=false
USER_CONFIGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--data)
            DATA_FILE="$2"
            shift 2
            ;;
        -n|--num-graphs)
            NUM_GRAPHS="$2"
            shift 2
            ;;
        -c|--config)
            USER_CONFIGS+=("$2")
            shift 2
            ;;
        --reset-configs)
            RESET_CONFIGS=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 处理配置
FINAL_CONFIGS=()
if [ "$RESET_CONFIGS" = true ]; then
    # 仅使用用户提供的配置
    FINAL_CONFIGS=("${USER_CONFIGS[@]}")
else
    # 合并默认配置和用户配置
    FINAL_CONFIGS=("${CONFIGS[@]}")
    
    # 用户配置覆盖默认配置
    for ((i=0; i<${#USER_CONFIGS[@]}; i++)); do
        if [ $i -lt ${#FINAL_CONFIGS[@]} ]; then
            FINAL_CONFIGS[$i]="${USER_CONFIGS[$i]}"
        else
            FINAL_CONFIGS+=("${USER_CONFIGS[$i]}")
        fi
    done
fi

# 限制配置数量与NUM_GRAPHS一致
if [ ${#FINAL_CONFIGS[@]} -gt $NUM_GRAPHS ]; then
    FINAL_CONFIGS=("${FINAL_CONFIGS[@]:0:$NUM_GRAPHS}")
elif [ ${#FINAL_CONFIGS[@]} -lt $NUM_GRAPHS ]; then
    echo "警告: 配置数量($NUM_GRAPHS)不足，使用可用的${#FINAL_CONFIGS[@]}个配置"
    NUM_GRAPHS=${#FINAL_CONFIGS[@]}
fi

# 构建命令行参数
CMD="./test_multi_graph_builder $DATA_FILE $NUM_GRAPHS"

for config in "${FINAL_CONFIGS[@]}"; do
    # 验证配置参数数量
    param_count=$(echo $config | wc -w)
    if [ $param_count -ne 5 ]; then
        echo "错误: 配置格式不正确 '$config'"
        echo "正确格式: \"nn_graph L R C output\" (5个参数)"
        exit 1
    fi
    
    # 添加到命令行
    CMD="$CMD $config"
done

# 显示将要执行的命令
echo "执行命令: $CMD"
echo "-------------------------------------------"

# 执行命令
eval $CMD
