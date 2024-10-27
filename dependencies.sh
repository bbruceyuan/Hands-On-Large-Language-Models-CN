
# python 环境相关
# step1: 初始化 conda 
conda activate /openbayes/home/envs/hands-on-llm/

# step2: install ipykernel
python -m ipykernel install --user --name hands-on-llm

# step3： 设置 huggingface 的 home
export HF_HOME=/openbayes/home/huggingface


# git 相关配置
# 添加这行来启动 SSH agenteval "$(ssh-agent -s)"  # 添加这行来启动 SSH agent
eval "$(ssh-agent -s)"  
chmod 600 /openbayes/home/.ssh/id_rsa
chmod 644 /openbayes/home/.ssh/id_rsa.pub
ssh-add /openbayes/home/.ssh/id_rsa

ssh -T git@github.com

# 设置 git 相关的 alias 到 .bashrc 中
# 好像暂时不需要这个了；

# 检查 ~/.cursor-server 是否存在，如果不存在则创建
# if [ ! -d ~/.cursor-server ]; then
#     mkdir ~/.cursor-server
# fi
# if [ ! -d ~/.vscode-server ]; then
#     mkdir ~/.vscode-server
# fi

# ln -s /openbayes/home/code-server/.cursor-server ~/.cursor-server
# ln -s /openbayes/home/code-server/.vscode-server ~/.vscode-server

## 方案2 设置cursor server
cp -r /openbayes/home/.cursor-server ~/.cursor-server
ln -s /openbayes/home/.cursor-server ~/.cursor-server