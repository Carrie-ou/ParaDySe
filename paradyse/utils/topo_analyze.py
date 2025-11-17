import subprocess
import re
import json
import platform


def run_command(command):
    """运行命令并返回输出"""
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {result.stderr}")
    return result.stdout


def parse_nvidia_smi_topo(output):
    """解析 nvidia-smi topo -m 的输出"""
    # 去除 ANSI 控制序列（如颜色标记）
    output = re.sub(r'\x1b\[[^m]*m', '', output)

    # 去除多余的空格和换行符
    lines = [line.strip() for line in output.split("\n") if line.strip()]

    # 提取表头
    header = lines[0].split("\t")
    gpu_ids = []
    for i, part in enumerate(header):
        if part.startswith("GPU") and "NUMA" not in part:
            gpu_ids.append(i)

    topology = []
    for line in lines[1:1 + len(gpu_ids)]:  # 忽略最后一行（Legend）
        parts = line.split("\t")
        gpu_id = parts[0]
        connections = {}
        for i in gpu_ids:
            connection = parts[i + 1]
            if connection != " X " and connection.strip():
                connected_gpu_id = header[i]
                connections[connected_gpu_id] = connection.strip()
        topology.append({
            "gpu_id": gpu_id,
            "connections": connections
        })

    return topology


def get_gpu_names():
    """获取所有 GPU 的名称"""
    output = run_command("nvidia-smi --query-gpu=name --format=csv,noheader")
    return [name.strip() for name in output.splitlines()]


def get_pcie_info():
    """获取 PCIe 连接信息（示例实现，可能需要调整）"""
    pcie_info = run_command("lspci | grep -i nvidia")
    return [line.strip() for line in pcie_info.splitlines()]


def dump_topo_info(file_path="./"):
    # 获取 GPU 名称
    gpu_names = get_gpu_names()
    gpu_name_map = {f"GPU{i}": name for i, name in enumerate(gpu_names)}

    # 获取拓扑信息
    topo_output = run_command("nvidia-smi topo -m")
    topo_data = parse_nvidia_smi_topo(topo_output)

    # 获取 PCIe 信息（示例）
    pcie_info = get_pcie_info()

    # 构建拓扑结果
    result = {
        "system_info": {
            "platform": platform.platform(),
            "os_version": platform.version(),
            "architecture": platform.architecture(),
        },
        "gpus": gpu_name_map,
        "topology": topo_data,
        "pcie_info": pcie_info
    }

    # 打印结果
    json.dump(result, open(file_path + "topo_info.json", "w"), indent=4)
    print(f"Topology information has been saved to {file_path}topo_info.json.")


if __name__ == "__main__":
    dump_topo_info()
