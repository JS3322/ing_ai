import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 1) PINN 모델 정의 (단순 MLP)
# -----------------------------
class PINN(nn.Module):
    def __init__(self, layers=[2, 32, 32, 1]):
        """
        layers: 예) [2, 32, 32, 1]
        입력차원 2: (x, t)
        출력차원 1: u(x,t)
        """
        super(PINN, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            in_dim = layers[i]
            out_dim = layers[i + 1]
            self.linears.append(nn.Linear(in_dim, out_dim))

    def forward(self, x, t):
        """
        x, t shape = (batch_size, 1)
        모델 입력을 (x, t) concat해서 예측
        """
        # 합치기
        X = torch.cat([x, t], dim=1)  # shape=[batch, 2]
        for i, layer in enumerate(self.linears[:-1]):
            X = torch.tanh(layer(X))
        X = self.linears[-1](X)
        return X  # shape=[batch, 1]


# -----------------------------
# 2) PDE(Heat Equation) Residual
# -----------------------------
def pde_residual(pinn_model, x, t, alpha=0.01):
    """
    열방정식: u_t = alpha * u_xx
    x, t: requires_grad_(True)
    """
    # 예측
    u = pinn_model(x, t)

    # u_t
    u_t = torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # u_x
    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # u_xx
    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    # PDE: u_t - alpha*u_xx = 0 -> residual
    residual = u_t - alpha * u_xx
    return residual


# -----------------------------
# 3) 예시 학습 루프
# -----------------------------
def train_pinn(pinn_model,
               X_col, Y_col,  # 내장(콜로케이션) 점 => PDE 잔차 학습용
               X_bc, Y_bc, U_bc,  # 경계조건 (or 초기조건) 점 => 값 고정
               X_data, T_data, U_data,  # 실제 관측(실험) 데이터
               alpha=0.01,
               epochs=2000,
               lr=1e-3,
               device='cpu'):
    """
    - X_col, Y_col: PDE residual loss를 계산할 (x, t) 샘플들 (collocation points)
    - X_bc, Y_bc, U_bc: 경계/초기조건
    - X_data, T_data, U_data: 실제 관측 데이터
    """
    optimizer = torch.optim.Adam(pinn_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 텐서 변환
    X_col_t = torch.tensor(X_col, dtype=torch.float32, requires_grad=True).to(device)
    Y_col_t = torch.tensor(Y_col, dtype=torch.float32, requires_grad=True).to(device)

    X_bc_t = torch.tensor(X_bc, dtype=torch.float32).to(device)
    Y_bc_t = torch.tensor(Y_bc, dtype=torch.float32).to(device)
    U_bc_t = torch.tensor(U_bc, dtype=torch.float32).to(device)

    X_data_t = torch.tensor(X_data, dtype=torch.float32).to(device)
    T_data_t = torch.tensor(T_data, dtype=torch.float32).to(device)
    U_data_t = torch.tensor(U_data, dtype=torch.float32).to(device)

    pinn_model.to(device)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # 1) PDE residual loss
        #    (x, t) in [collocation points]
        res = pde_residual(pinn_model, X_col_t, Y_col_t, alpha=alpha)
        loss_pde = torch.mean(res ** 2)

        # 2) BC/IC loss (경계 or 초기조건)
        pred_bc = pinn_model(X_bc_t, Y_bc_t)
        loss_bc = loss_fn(pred_bc, U_bc_t)

        # 3) Data loss (실험/시뮬레이션 결과)
        pred_data = pinn_model(X_data_t, T_data_t)
        loss_data = loss_fn(pred_data, U_data_t)

        # 전체 loss
        loss = loss_pde + loss_bc + loss_data
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == epochs:
            print(f"[Epoch {epoch}/{epochs}] "
                  f"PDE={loss_pde.item():.4e}, BC={loss_bc.item():.4e}, Data={loss_data.item():.4e}, Total={loss.item():.4e}")


# -----------------------------
# 4) 데모 실행
# -----------------------------
def demo_pinn_heat_equation():
    # 0) parameter
    alpha = 0.01
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # 1) Collocation points (PDE domain)
    #    x ~ [0,1], t ~ [0,1]
    N_col = 2000
    x_col = np.random.rand(N_col, 1)  # [0,1]
    t_col = np.random.rand(N_col, 1)  # [0,1]

    # 2) BC/IC
    #    예) u(0,t)=0, u(1,t)=0, u(x,0)=sin(pi*x)
    #    -> 여기선 간단히 x=0, x=1, 그리고 t=0 부분 추출
    x_bc0 = np.zeros((50, 1))  # x=0, t in [0,1]
    t_bc0 = np.linspace(0, 1, 50).reshape(-1, 1)
    u_bc0 = np.zeros((50, 1))

    x_bc1 = np.ones((50, 1))  # x=1, t in [0,1]
    t_bc1 = np.linspace(0, 1, 50).reshape(-1, 1)
    u_bc1 = np.zeros((50, 1))

    # 초기조건 x in [0,1], t=0
    x_bc2 = np.linspace(0, 1, 50).reshape(-1, 1)
    t_bc2 = np.zeros((50, 1))
    u_bc2 = np.sin(np.pi * x_bc2)

    # BC/IC 합침
    X_bc = np.concatenate([x_bc0, x_bc1, x_bc2], axis=0)
    Y_bc = np.concatenate([t_bc0, t_bc1, t_bc2], axis=0)
    U_bc = np.concatenate([u_bc0, u_bc1, u_bc2], axis=0)

    # 3) 실제 관측 데이터 (예: 일부 시점에서 측정된 u(x,t))
    #    여기서는 인위적으로 정답 해석해(analytic) 생성
    #    heat eq 해석해는 u(x,t)=exp(-pi^2*alpha*t)*sin(pi*x) (1차원 기준)
    def true_solution(x, t, alpha=0.01):
        return np.exp(- (np.pi ** 2) * alpha * t) * np.sin(np.pi * x)

    N_data = 200
    x_data = np.random.rand(N_data, 1)
    t_data = np.random.rand(N_data, 1)
    u_data = true_solution(x_data, t_data, alpha)

    # 4) PINN 모델
    pinn = PINN(layers=[2, 32, 32, 32, 1])

    # 5) 학습
    train_pinn(pinn,
               X_col=x_col, Y_col=t_col,
               X_bc=X_bc, Y_bc=Y_bc, U_bc=U_bc,
               X_data=x_data, T_data=t_data, U_data=u_data,
               alpha=alpha,
               epochs=2000,
               lr=1e-3,
               device=device)

    # 6) 결과 확인
    #    x=0.5 고정, t in [0,1]에 대한 예측
    t_test = np.linspace(0, 1, 50).reshape(-1, 1).astype(np.float32)
    x_test = 0.5 * np.ones_like(t_test)
    t_torch = torch.tensor(t_test, requires_grad=False).to(device)
    x_torch = torch.tensor(x_test, requires_grad=False).to(device)

    pinn.eval()
    with torch.no_grad():
        pred = pinn(x_torch, t_torch).cpu().numpy()
    exact = true_solution(x_test, t_test, alpha)

    plt.plot(t_test, exact, 'k--', label='Exact')
    plt.plot(t_test, pred, 'r-', label='PINN')
    plt.xlabel("t"), plt.ylabel("u(x=0.5,t)")
    plt.legend()
    plt.title("PINN Heat Equation (x=0.5)")
    plt.show()


if __name__ == "__main__":
    demo_pinn_heat_equation()
