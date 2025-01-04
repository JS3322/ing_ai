import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# (중요) PyTorch Geometric 관련 임포트
# pip install torch-geometric 등으로 설치 필요
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv  # 예: GCN 레이어

###############################################################################
# 1) 그래프 데이터 구성 (예시)
###############################################################################
def create_example_graph():
    """
    예시:
      - 노드가 6개 (공정 장비 6대로 가정)
      - 노드 피처는 임의의 2차원 벡터
      - 엣지는 아래 adjacency를 통해 정의
      - 노드 라벨: 2가지 클래스(0 또는 1)로 가정
    """
    # 노드 수
    num_nodes = 6

    # 간단한 adjacency: (i->j 엣지들을 정의)
    # 예) 장비 0과 1이 연결, 1과 2가 연결 등
    edges = [
        [0, 1],
        [1, 2],
        [1, 3],
        [2, 4],
        [2, 5],
        [3, 4]
    ]
    # edges를 PyG 포맷 (2 x E) 텐서로
    edge_index = torch.tensor(edges, dtype=torch.long).t()  # shape=[2, E]

    # 노드 피처 (여기선 임의로 2차원 float)
    # 실제로는 장비별 센서 값 등
    x = torch.tensor([
        [1.0, 2.0],   # Node 0
        [0.5, 1.5],   # Node 1
        [2.2, 0.1],   # Node 2
        [1.0, 1.0],   # Node 3
        [2.5, 2.5],   # Node 4
        [0.8, 0.2],   # Node 5
    ], dtype=torch.float)

    # 노드 라벨 (0/1 이진분류 예시)
    # 예: 장비의 정상/오류 상태, 혹은 결함 타입 등
    y = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long)

    # PyG의 Data 객체 생성
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

###############################################################################
# 2) GNN 모델 (GCNConv)
###############################################################################
class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 1) 첫 번째 GCN 레이어
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 2) 두 번째 GCN 레이어
        x = self.conv2(x, edge_index)
        # 여기서는 node-level classification 가정
        return x

###############################################################################
# 3) 학습/평가 루프
###############################################################################
def train_gnn(model, data, epochs=100, lr=1e-2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        out = model(data)       # shape = [num_nodes, num_classes]
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss={loss.item():.4f}")

def test_gnn(model, data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)  # [num_nodes]
    correct = (pred == data.y).sum().item()
    acc = correct / data.y.size(0)
    print(f"Accuracy: {acc*100:.2f}%")

###############################################################################
# 4) 메인 실행
###############################################################################
def main():
    # 1) 그래프 생성
    data = create_example_graph()
    print("Graph data:", data)
    # Data(x=[6, 2], edge_index=[2, 6], y=[6])

    # 2) GCN 모델 초기화
    # in_channels=2 (노드 피처 차원), hidden_channels=4, out_channels=2 (이진분류)
    model = GCNModel(in_channels=2, hidden_channels=4, out_channels=2)

    # 3) 학습
    train_gnn(model, data, epochs=100, lr=0.01)

    # 4) 평가
    test_gnn(model, data)

if __name__ == "__main__":
    main()
