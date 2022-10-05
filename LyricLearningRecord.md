## Git init
先在github remote建repo 
rename main -> master
git init
git remote add origin git@github.com:Lyric98/xxxxxxxx.git
git remote -v (检查是否连上)
git pull origin master
git branch -a

## 环境
conda create -n your_env_name python=x.x Python创建虚拟环境

conda activate RL

conda deactivate (退出该环境)

conda env list 查看已有环境list

## vscode 打开项目
comment shift P 输入 “interpreter” 选中所属环境