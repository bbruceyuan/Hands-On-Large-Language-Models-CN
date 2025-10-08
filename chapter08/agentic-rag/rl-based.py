import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class TokenStep:
    token_id: int
    token_text: str
    log_prob: float
    position: int


@dataclass
class Trajectory:
    query: str
    token_steps: List[TokenStep]
    generated_text: str
    reward: float
    final_answer: str


class SearchEngine:
    """搜索引擎"""
    def __init__(self):
        self.knowledge_base = {
            "python": "Python is a high-level programming language known for its simplicity and readability.",
            "machine learning": "Machine learning is a subset of AI that enables computers to learn from experience.",
            "neural networks": "Neural networks are computing systems inspired by biological neural networks.",
            "deep learning": "Deep learning is a subset of machine learning using artificial neural networks.",
            "transformer": "Transformers are neural network architectures using self-attention mechanisms.",
            "reinforcement learning": "Reinforcement learning involves agents learning through environment interaction."
        }
    
    def search(self, query: str) -> str:
        query_lower = query.lower().strip()
        for key, value in self.knowledge_base.items():
            if key in query_lower:
                return value
        return f"No information found for: {query}"


class SearchR1GRPO:
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small", lr: float = 3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 添加特殊 tokens
        special_tokens = ["<think>", "</think>", "<search>", "</search>", "<information>", "</information>", "<answer>", "</answer>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        
        # GRPO 参数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.beta = 0.01  # KL 散度系数
        self.delta = 0.01  # KL 目标值
        self.clip_epsilon = 0.2  # PPO 裁剪参数
        self.target_kl = self.delta * 1.5
        
        self.search_engine = SearchEngine()
        
        # 特殊 token IDs
        self.special_token_ids = {
            "<think>": self.tokenizer.convert_tokens_to_ids("<think>"),
            "</think>": self.tokenizer.convert_tokens_to_ids("</think>"),
            "<search>": self.tokenizer.convert_tokens_to_ids("<search>"),
            "</search>": self.tokenizer.convert_tokens_to_ids("</search>"),
            "<information>": self.tokenizer.convert_tokens_to_ids("<information>"),
            "</information>": self.tokenizer.convert_tokens_to_ids("</information>"),
            "<answer>": self.tokenizer.convert_tokens_to_ids("<answer>"),
            "</answer>": self.tokenizer.convert_tokens_to_ids("</answer>")
        }
    
    def generate_trajectory(self, query: str, max_tokens: int = 150) -> Trajectory:
        """生成轨迹 - 每个 token 作为一个动作"""
        self.model.eval()
        
        # 初始 prompt
        prompt = f"you should put think in <think> and search in <search>. then answer in <answer>. Question: {query}\nLet me think step by step. \n<think>"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        token_steps = []
        generated_tokens = []
        current_text = prompt
        
        with torch.no_grad():
            for step in range(max_tokens):
                # 前向传播
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]  # 最后一个位置的 logits
                
                # 计算概率分布
                probs = F.softmax(logits, dim=-1)
                
                # 采样下一个 token
                token_dist = torch.distributions.Categorical(probs)
                next_token_id = token_dist.sample()
                log_prob = token_dist.log_prob(next_token_id).item()
                
                # 解码 token
                token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
                
                # 记录步骤
                token_step = TokenStep(
                    token_id=next_token_id.item(),
                    token_text=token_text,
                    log_prob=log_prob,
                    position=step
                )
                token_steps.append(token_step)
                generated_tokens.append(next_token_id.item())
                
                # 更新输入
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                current_text += token_text
                
                if next_token_id.item() == self.special_token_ids["</search>"]:
                    search_query = self.extract_search_query(current_text)
                    if search_query:
                        search_result = self.search_engine.search(search_query)
                        info_text = f"\n<information>{search_result}</information>\n"
                        info_tokens = self.tokenizer.encode(info_text, add_special_tokens=False)
                        
                        # 将搜索结果添加到输入中
                        info_tensor = torch.tensor(info_tokens).unsqueeze(0).to(self.device)
                        input_ids = torch.cat([input_ids, info_tensor], dim=1)
                        current_text += info_text
                
                # 检查是否遇到 </answer> 结束生成
                if next_token_id.item() == self.special_token_ids["</answer>"]:
                    break
                
                # 检查其他结束条件
                if (next_token_id.item() == self.tokenizer.eos_token_id or 
                    step >= max_tokens - 1):
                    break
        
        # 生成的完整文本
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
        
        # 提取最终答案
        final_answer = self.extract_final_answer(current_text)
        
        return Trajectory(
            query=query,
            token_steps=token_steps,
            generated_text=generated_text,
            reward=0.0,  # 稍后计算
            final_answer=final_answer
        )
    
    def extract_search_query(self, text: str) -> str:
        """从文本中提取搜索查询"""
        # 查找最后一个 <search>...</search> 块
        pattern = r'<search>(.*?)</search>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return ""
    
    def extract_final_answer(self, text: str) -> str:
        """提取最终答案 - 从 <answer>...</answer> 中提取"""
        # 查找 <answer>...</answer> 块
        pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()  # 取最后一个匹配（应该只有一个）
        return ""
    
    def compute_reward(self, trajectory: Trajectory, ground_truth: str) -> float:
        """
        计算奖励 - 包含答案正确性和格式正确性
        答案正确: 完全一致得1分，否则0分
        格式正确: 符合格式得0.1分，否则-1分
        """
        # 1. 格式正确性检查
        format_reward = self.check_format_correctness(trajectory.generated_text)
        
        # 2. 答案正确性检查
        answer_reward = self.check_answer_correctness(trajectory.final_answer, ground_truth)
        
        # 总奖励 = 答案奖励 + 格式奖励
        total_reward = answer_reward + format_reward
        
        return total_reward
    
    def check_format_correctness(self, generated_text: str) -> float:
        """
        检查格式正确性
        要求:
        1. 可以有多轮 think/search/information 循环
        2. <answer></answer> 只能出现一次，且在最末尾
        3. 所有标签必须成对出现
        """
        # 检查 answer 标签
        answer_start_count = generated_text.count("<answer>")
        answer_end_count = generated_text.count("</answer>")
        
        # answer 标签必须恰好出现一次
        if answer_start_count != 1 or answer_end_count != 1:
            return -1.0  # answer 标签数量错误
        
        # 检查 answer 是否在最末尾
        answer_start_pos = generated_text.rfind("<answer>")
        answer_end_pos = generated_text.rfind("</answer>")
        
        if answer_start_pos == -1 or answer_end_pos == -1 or answer_start_pos >= answer_end_pos:
            return -1.0  # answer 标签位置错误
        
        # 检查 answer 后面是否还有其他内容（除了空白字符）
        after_answer = generated_text[answer_end_pos + len("</answer>"):].strip()
        if after_answer:
            return -1.0  # answer 后面还有内容
        
        # 检查其他标签的配对
        tag_pairs = [
            ("<think>", "</think>"),
            ("<search>", "</search>"),
            ("<information>", "</information>")
        ]
        
        for start_tag, end_tag in tag_pairs:
            start_count = generated_text.count(start_tag)
            end_count = generated_text.count(end_tag)
            if start_count != end_count:
                return -1.0  # 标签不配对
        
        # 检查是否至少有一个 think 标签
        if generated_text.count("<think>") == 0:
            return -1.0  # 缺少必需的 think 标签
        
        return 0.1  # 格式完全正确
    
    def check_answer_correctness(self, final_answer: str, ground_truth: str) -> float:
        """
        检查答案正确性
        完全一致得1分，否则0分
        直接比较，不进行文本标准化
        """
        if not final_answer or not ground_truth:
            return 0.0
        
        # 直接比较，完全匹配
        if final_answer == ground_truth:
            return 1.0
        else:
            return 0.0
    
    def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        """计算相对优势"""
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean_reward = torch.mean(rewards_tensor)
        std_reward = torch.std(rewards_tensor) + 1e-8
        advantages = (rewards_tensor - mean_reward) / std_reward
        return advantages
    
    def compute_kl_divergence(self, old_log_probs: torch.Tensor, new_log_probs: torch.Tensor) -> torch.Tensor:
        """计算 KL 散度"""
        return torch.mean(torch.exp(old_log_probs) * (old_log_probs - new_log_probs))
    
    def recompute_log_probs(self, trajectories: List[Trajectory]) -> List[torch.Tensor]:
        """重新计算轨迹的对数概率"""
        self.model.eval()
        all_log_probs = []
        
        with torch.no_grad():
            for traj in trajectories:
                # 重建输入序列
                prompt = f"Question: {traj.query}\nLet me think step by step.\n<think>"
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                
                log_probs = []
                
                for token_step in traj.token_steps:
                    # 前向传播
                    outputs = self.model(input_ids)
                    logits = outputs.logits[0, -1, :]
                    
                    # 计算对数概率
                    log_prob = F.log_softmax(logits, dim=-1)[token_step.token_id]
                    log_probs.append(log_prob)
                    
                    # 更新输入
                    next_token = torch.tensor([[token_step.token_id]]).to(self.device)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                
                all_log_probs.append(torch.stack(log_probs))
        
        return all_log_probs
    
    def update_policy(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """GRPO 策略更新"""
        if not trajectories:
            return {"loss": 0.0, "kl_div": 0.0}
        
        self.model.train()
        
        # 计算奖励和优势
        rewards = [traj.reward for traj in trajectories]
        advantages = self.compute_advantages(rewards)
        
        # 获取旧的对数概率
        old_log_probs_list = []
        for traj in trajectories:
            old_probs = torch.tensor([step.log_prob for step in traj.token_steps])
            old_log_probs_list.append(old_probs)
        
        # 多轮优化
        total_loss = 0.0
        final_kl_div = 0.0
        
        for epoch in range(3):
            # 重新计算当前策略的对数概率
            new_log_probs_list = self.recompute_log_probs(trajectories)
            
            epoch_loss = 0.0
            epoch_kl = 0.0
            
            for i, (traj, old_log_probs, new_log_probs) in enumerate(
                zip(trajectories, old_log_probs_list, new_log_probs_list)
            ):
                if len(old_log_probs) != len(new_log_probs):
                    continue
                
                # 计算概率比
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # 扩展优势到所有 token
                traj_advantage = advantages[i].repeat(len(ratio))
                
                # PPO 裁剪目标
                surr1 = ratio * traj_advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * traj_advantage
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # KL 散度
                kl_div = self.compute_kl_divergence(old_log_probs, new_log_probs)
                
                # 总损失
                loss = policy_loss + self.beta * kl_div
                epoch_loss += loss
                epoch_kl += kl_div
            
            # 平均损失
            if len(trajectories) > 0:
                epoch_loss /= len(trajectories)
                epoch_kl /= len(trajectories)
            
            # 反向传播
            self.optimizer.zero_grad()
            epoch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            # 动态调整 beta
            if epoch_kl > self.target_kl:
                self.beta *= 2
            elif epoch_kl < self.delta / 1.5:
                self.beta /= 2
            
            # KL 散度过大时提前终止
            if epoch_kl > self.delta * 2:
                break
            
            total_loss = epoch_loss.item()
            final_kl_div = epoch_kl.item()
        
        return {
            "loss": total_loss,
            "kl_div": final_kl_div,
            "avg_reward": np.mean(rewards),
            "beta": self.beta
        }
    
    def train_step(self, queries: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """执行一步训练"""
        # 生成轨迹
        trajectories = []
        for query, truth in zip(queries, ground_truths):
            trajectory = self.generate_trajectory(query, max_tokens=100)
            trajectory.reward = self.compute_reward(trajectory, truth)
            trajectories.append(trajectory)
        
        # 更新策略
        metrics = self.update_policy(trajectories)
        
        # 添加统计信息
        avg_tokens = np.mean([len(traj.token_steps) for traj in trajectories])
        search_count = sum(1 for traj in trajectories if "<search>" in traj.generated_text)
        
        metrics.update({
            "avg_tokens": avg_tokens,
            "search_trajectories": search_count / len(trajectories) if trajectories else 0
        })
        
        return metrics

def create_training_data() -> Tuple[List[str], List[str]]:
    """创建训练数据"""
    queries = [
        "What is Python?",
        "Explain machine learning",
        "What are neural networks?",
        "What is deep learning?",
        "Explain transformers"
    ]
    
    ground_truths = [
        "Python is a programming language",
        "Machine learning is AI subset",
        "Neural networks are computing systems",
        "Deep learning uses neural networks",
        "Transformers use attention mechanisms"
    ]
    
    return queries, ground_truths

def main():
    """主训练循环"""
    print("初始化 Search-R1 GRPO (Token-level) 训练器...")
    trainer = SearchR1GRPO()
    
    # 创建训练数据
    queries, ground_truths = create_training_data()
    
    print("开始训练...")
    num_epochs = 15
    
    for epoch in range(num_epochs):
        metrics = trainer.train_step(queries, ground_truths)
        
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  KL Div: {metrics['kl_div']:.4f}")
        print(f"  Avg Reward: {metrics['avg_reward']:.4f}")
        print(f"  Avg Tokens: {metrics['avg_tokens']:.1f}")
        print(f"  Search Rate: {metrics['search_trajectories']:.2f}")
        print(f"  Beta: {metrics['beta']:.4f}")
        print("-" * 50)
    
    # 测试
    print("\n测试训练后的模型:")
    test_query = "What is machine learning?"
    trajectory = trainer.generate_trajectory(test_query, max_tokens=80)
    
    print(f"Query: {test_query}")
    print(f"Generated: {trajectory.generated_text}")
    print(f"Final Answer: {trajectory.final_answer}")
    print(f"Tokens: {len(trajectory.token_steps)}")
    print(f"Reward: {trainer.compute_reward(trajectory, 'Machine learning is AI'):.1f}")

if __name__ == "__main__":
    main()