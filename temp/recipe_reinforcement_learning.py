import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# 1) 공정 환경 (간단 예시)
# -----------------------
class ProcessEnv:
    """
    상태: [temperature, thickness_error]
    행동: 0 -> temp - 1, 1 -> temp + 1
    목표: thickness_error를 0에 가깝게 유지
    """
    def __init__(self):
        self.temperature = 50.0  # 초기 온도
        self.thickness_error = 10.0  # 초기 두께 오차
        self.step_count = 0
        self.max_steps = 50

    def reset(self):
        self.temperature = 50.0
        self.thickness_error = 10.0
        self.step_count = 0
        return self._get_state()

    def step(self, action):
        # action: 0 -> temp - 1, 1 -> temp + 1
        if action == 0:
            self.temperature -= 1.0
        else:
            self.temperature += 1.0

        # 어떤 식으로든 thickness_error 업데이트 (가상 로직)
        # 예: temp가 70 근처일 때 thickness_error가 감소, 너무 높아지면 증가
        # 아주 단순한 함수로 예시
        ideal_temp = 70.0
        diff = abs(self.temperature - ideal_temp)
        self.thickness_error = diff  # 단순히 temp-ideal 의 절댓값으로 가정

        # 보상 계산: 두께오차가 작을수록 보상↑
        reward = -self.thickness_error

        self.step_count += 1
        done = (self.step_count >= self.max_steps)
        next_state = self._get_state()
        return next_state, reward, done

    def _get_state(self):
        return np.array([self.temperature, self.thickness_error], dtype=np.float32)

# -----------------------
# 2) DQN 네트워크 정의
# -----------------------
class DQN(nn.Module):
    def __init__(self, state_dim=2, hidden_dim=64, action_dim=2):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------
# 3) 에이전트
# -----------------------
class DQNAgent:
    def __init__(self, state_dim=2, action_dim=2,
                 lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_dim = action_dim

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dqn = DQN(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.target_dqn = DQN(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())  # 초기엔 동일

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = []     # (s, a, r, s_next, done)
        self.batch_size = 32
        self.capacity = 10000
        self.update_target_steps = 50
        self.learn_step_counter = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.dqn(state_t)
                action = torch.argmax(q_values, dim=1).item()
            return action

    def store_transition(self, s, a, r, s_next, done):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append((s, a, r, s_next, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_next, done = zip(*batch)

        s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.long, device=self.device).unsqueeze(1)
        r_t = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s_next_t = torch.tensor(s_next, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_values = self.dqn(s_t).gather(1, a_t)
        # Q(s',a') (Target)
        with torch.no_grad():
            q_next = self.target_dqn(s_next_t).max(dim=1, keepdim=True)[0]
            q_target = r_t + self.gamma * q_next * (1 - done_t)

        loss = self.criterion(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_target_steps == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# -----------------------
# 4) 학습 루프
# -----------------------
def main():
    env = ProcessEnv()
    agent = DQNAgent()

    num_episodes = 200
    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward

        if (ep+1) % 20 == 0:
            print(f"[Episode {ep+1}] Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    # 학습 완료 후 시험
    test_state = env.reset()
    done = False
    while not done:
        action = agent.select_action(test_state)
        next_state, reward, done = env.step(action)
        test_state = next_state
        # (실제 현장에선 온도, 두께오차 변화를 확인)
    print("Test run finished.")

if __name__ == "__main__":
    main()
