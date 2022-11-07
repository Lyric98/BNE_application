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
git 撤销commit： https://www.cnblogs.com/lfxiao/p/9378763.html 
```
git reset --soft HEAD^ #HEAD^的意思是上一个版本，也可以写成HEAD~1
                      #如果你进行了2次commit，想都撤回，可以使用HEAD~2
git log 查询commit
```

## 环境
conda create -n your_env_name python=x.x Python创建虚拟环境

conda activate RL

conda deactivate (退出该环境)

conda env list 查看已有环境list

## vscode 打开项目
comment shift P 输入 “interpreter” 选中所属环境

jupyter notebook 转换成 python script
```
jupyter nbconvert --to script
```

## Linux 文件相关
Linux下删除指定文件夹下指定后缀名的文件: 命令有点危险，可以先执行前半段，看看是不是你要删除的文件, 然后再整条执行

```
find . -name "*.out"  
find . -name "*.out"| xargs rm
```


## 引用同地址的文件出现 module not found error
```
pip install -e .
(fsvi) liyanran@liyanrandeMBP function-space-variational-inference-yanran % pip install -e .
Obtaining file:///Users/liyanran/Desktop/Andrew/function-space-variational-inference-yanran
  Preparing metadata (setup.py) ... done
Installing collected packages: fsvi
  Running setup.py develop for fsvi
Successfully installed fsvi-0.1
```
### toy objective function --BO
```
cd ../..
which python #确保运行的是python3
python run_base.py --data_training bayesian_optimization --model mlp_fsvi --architecture fc_100_100 --activation tanh --learning_rate 1e-3 --optimizer adam --batch_size 0 --prior_mean 0 --prior_cov 4 --prior_cov_offset 1 --prior_type bnn_induced --n_inducing_inputs 100 --n_marginals 1 --inducing_inputs_bound -10 10 --inducing_input_type uniform_rand_tdvi --kl_scale none --n_samples 5 --n_samples_eval 100 --logging_frequency 1000 --seed 0 --debug --save --save_path tmp --feature_map_jacobian --feature_map_type learned_grad --grad_flow_jacobian --full_cov --prior_mean_init --noise_std 0.01 --epochs 10000
```

