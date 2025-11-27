#!/bin/bash
# 监控基准测试进度

LOG_FILE="/home/lzx/Llama-diffusion/benchmark_run.log"
RESULTS_DIR="/home/lzx/Llama-diffusion/benchmark_results"

echo "=========================================="
echo "SDAR Benchmark Progress Monitor"
echo "=========================================="

# 检查进程是否在运行
PID=$(pgrep -f "benchmark_sdar.py")
if [ -n "$PID" ]; then
    echo "✅ Benchmark is running (PID: $PID)"
else
    echo "❌ Benchmark is not running"
fi

echo ""
echo "--- Latest log entries ---"
tail -30 "$LOG_FILE" 2>/dev/null | grep -E "(Evaluating|Result|Progress|Testing|Error|completed)"

echo ""
echo "--- Current results ---"
LATEST_DIR=$(ls -td "$RESULTS_DIR"/*/ 2>/dev/null | head -1)
if [ -n "$LATEST_DIR" ]; then
    echo "Results directory: $LATEST_DIR"
    if [ -f "$LATEST_DIR/results.json" ]; then
        echo "Datasets completed:"
        cat "$LATEST_DIR/results.json" | python3 -c "
import json, sys
data = json.load(sys.stdin)
configs = set()
datasets = set()
for r in data:
    configs.add(r['config_name'])
    datasets.add(r['dataset_name'])
print(f'  Configs: {len(configs)}')
print(f'  Datasets: {len(datasets)}')
print(f'  Total runs: {len(data)}')
for r in data:
    print(f\"    {r['dataset_name']:15} {r['config_name']:20} Acc: {r['accuracy']:.2%}\")
"
    fi
fi

