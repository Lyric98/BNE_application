## Git init
先在github remote建repo
```
rename main -> master
git init
git remote add origin git@github.com:Lyric98/xxxxxxxx.git
git remote -v (检查是否连上)
git pull origin master
git branch -a
```

## 环境
conda create -n your_env_name python=x.x Python创建虚拟环境

conda activate RL

conda deactivate (退出该环境)

conda env list 查看已有环境list

## vscode 打开项目
comment shift P 输入 “interpreter” 选中所属环境

## 引用同地址的文件出现 module not found error
### pip install -e .
(fsvi) liyanran@liyanrandeMBP function-space-variational-inference-yanran % pip install -e .
Obtaining file:///Users/liyanran/Desktop/Andrew/function-space-variational-inference-yanran
  Preparing metadata (setup.py) ... done
Installing collected packages: fsvi
  Running setup.py develop for fsvi
Successfully installed fsvi-0.1