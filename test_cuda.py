import torch
import sys

def test_cuda():
    print("Python版本:", sys.version)
    print("PyTorch版本:", torch.__version__)
    print("\nCUDA是否可用:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("CUDA版本:", torch.version.cuda)
        print("当前设备:", torch.cuda.current_device())
        print("设备数量:", torch.cuda.device_count())
        print("设备名称:", torch.cuda.get_device_name(0))
        
        # 测试GPU内存分配
        try:
            x = torch.rand(1000, 1000).cuda()
            print("\nGPU内存分配测试成功")
            del x
            torch.cuda.empty_cache()
        except Exception as e:
            print("\nGPU内存分配测试失败:", str(e))
    else:
        print("\n警告: CUDA不可用，可能的原因：")
        print("1. CUDA工具包未正确安装")
        print("2. PyTorch未安装CUDA版本")
        print("3. CUDA版本与PyTorch版本不匹配")

if __name__ == '__main__':
    test_cuda()