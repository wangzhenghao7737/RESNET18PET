# 遇到的问题

## 租显卡

平台：www.autodl.com

* 镜像选择
  * 基础镜像-PyTorch-2.0.0

## vscode远程连接

```
ctrl+shift+~打开终端
```

## FileZilla文件上传

* 数据集建议先压缩，后解压

## 环境配置

* anaconda 是否需要init，能否直接进入激活环境

  * ```
    遇到错误
    CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'
    
    原因
    这是因为当前 shell 没有初始化 Conda 的环境。
    
    方法
    1查找conda安装路径
    sudo find / -name "conda" 2>/dev/null | grep bin/conda
    2重新加载shell配置
    source ~/.bashrc
    3激活环境
    conda activate pet
    
    实际解决
    法1
    找到conda路径：/root/miniconda3
    初始化conda：/root/miniconda3/bin/conda init bash
    重新加载shell配置
    source ~/.bashrc
    激活环境
    conda activate pet
    
    法2
    直接运行source /root/miniconda3/bin/activate pet
    
    
    ```

    

* 安装pytorch
  * 在pythorch官网选择
    * conda v1.10.1
  * 国内服务器安装conda，不建议-c conda-forge
* 安装其他的库
  * tqdm,pandas等
* 降级numpy库（2.0不支持）

## python代码跨平台

* 现在本地运行一遍，检查是否有代码问题

  * print打印语句出错

* 路径代码是否不兼容linux

  * 注意路径写法

  

